// Owns the AlphaZero engine: the wasm module holding Amcts2 and the position,
// and the onnxruntime-web session that evaluates the residual network on the
// GPU.
//
// The split is the whole point of this file. Amcts2 runs in wasm because it is
// C++ and unmodified; the network runs in JavaScript because WebGPU has no C++
// API. They meet at Module.azRun, which the wasm side calls through EM_ASYNC_JS
// and which suspends the wasm stack until ort resolves. That is why the module
// is linked with -sASYNCIFY and why _mgy_az_bot_move must be awaited.
//
// Two traps carried over from worker.js, both still live here:
//   - never cache a heap view. ALLOW_MEMORY_GROWTH detaches HEAPF32/HEAPU8 on
//     every grow, and a search allocates. Re-read mod.HEAPF32 on each use.
//   - _mgy_az_snapshot() is not a pure getter. It rewrites the struct from the
//     current position and returns its address; call it on every read.

'use strict';

// The loader is tiny (67 KB); it pulls its own .wasm/.mjs sidecars from the same
// directory, which is why wasmPaths is pinned to the CDN dist folder. To run
// fully offline, vendor onnxruntime-web@1.27.0/dist/ next to this file and point
// both the importScripts URL and wasmPaths at it.
const ORT_VERSION = '1.27.0';
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

// --- async bridge -----------------------------------------------------------
//
// Two builds of the same C++ exist, differing only in how Amcts2's synchronous
// search is suspended while onnxruntime-web resolves:
//
//   asyncify  the wasm is rewritten to unwind and rewind its own stack. Works
//             in every browser. Costs a stack copy on each of the ~150 suspends
//             per move, and more than doubles the module.
//   jspi      the VM switches stacks natively. Chrome 137+, Firefox 139+.
//             Safari has not shipped it yet.
//
// Measured on a 3060 Ti, same positions, batch 8, 3 s per move:
//
//   speed  identical - 468 ev/s (jspi) against 467 (asyncify), well inside the
//          run-to-run spread. Essentially the whole move is spent inside
//          session.run, so the bridge overhead is invisible next to a ~25 ms GPU
//          call. Do not expect JSPI to make the bot search deeper.
//   size   171 KB against 348 KB of wasm, plus 16 KB against 22 KB of glue. That
//          is the entire practical win, and only one of the two is ever fetched.
//
// So JSPI is kept as the default because it is smaller and is where the platform
// is going, not because it is faster here. If a future workload becomes
// CPU-bound in the tree rather than GPU-bound, re-measure before assuming.
//
// Detection matches what emscripten itself asserts on. That assert lives behind
// ASSERTIONS, which Release compiles out, so on an unsupported browser the JSPI
// build would fail obscurely rather than loudly - this check is what keeps the
// failure clean, and it runs before the glue is even fetched.
//
// ?bridge=jspi / ?bridge=asyncify forces one, so the two can be compared on the
// same machine without a rebuild. app.js appends it to the worker URL, since a
// worker does not inherit the page's query string.
const BRIDGE_OVERRIDE = new URLSearchParams(self.location.search).get('bridge');
const JSPI_AVAILABLE = 'Suspending' in WebAssembly && 'promising' in WebAssembly;
const USE_JSPI = BRIDGE_OVERRIDE === 'jspi' ? true
  : BRIDGE_OVERRIDE === 'asyncify' ? false
    : JSPI_AVAILABLE;
const BRIDGE = USE_JSPI ? 'jspi' : 'asyncify';

importScripts(`${ORT_DIST}ort.webgpu.min.js`);
importScripts('az_snapshot.js');
// Only one is ever loaded; both define self.createMigoyugoAz.
importScripts(USE_JSPI ? 'migoyugo_az_jspi.js' : 'migoyugo_az.js');

let mod = null;
let session = null;
let movePtr = 0; // scratch for load_moves, allocated once
let snapLen = 0;
let inputName = null;
let probsName = null;
let wdlName = null;
let provider = 'unknown';
// Set once a search has completed. Until then a failure may mean the chosen
// bridge does not really work here, which is recoverable by reloading on the
// other one; after that, a failure is a genuine error.
let searchSucceeded = false;

const OK = 0;
const ERRORS = {
  '-1': 'no model loaded',
  '-2': 'square out of range',
  '-3': 'illegal move',
  '-4': 'the game is already over',
  '-7': 'nothing to undo',
};
const describe = (code) => ERRORS[String(code)] || `error ${code}`;

// Must match kMaxPlies in migoyugo_az_wasm.cpp. Not 64: promotion recycles
// squares, so a game outruns the board size.
const MAX_PLIES = 255;

const OBSERVATION_SIZE = 4 * 8 * 8;
const N_ACTIONS = 64;

function snapshotBuffer() {
  const p = mod._mgy_az_snapshot();
  return mod.HEAPU8.slice(p, p + snapLen).buffer;
}

function probsArray() {
  const p = mod._mgy_az_probs();
  return Array.from(mod.HEAPF32.subarray(p >> 2, (p >> 2) + N_ACTIONS));
}

function sendState(epoch, extra = {}) {
  const snapshot = snapshotBuffer();
  postMessage({ type: 'state', epoch, snapshot, ...extra }, [snapshot]);
}

// The bridge the wasm side calls once per MCTS batch. Pointers arrive as byte
// offsets into linear memory; >> 2 converts them to Float32Array indices, which
// is valid because every buffer involved is a std::vector<float>.
async function azRun(obsPtr, nStates, probsPtr, wdlPtr) {
  const base = obsPtr >> 2;
  // slice(), not subarray(): ort keeps the buffer while it uploads, and a view
  // into wasm memory can be detached by a growth in the meantime.
  const observations = mod.HEAPF32.slice(base, base + nStates * OBSERVATION_SIZE);

  const feeds = {};
  feeds[inputName] = new ort.Tensor('float32', observations, [nStates, 4, 8, 8]);
  const output = await session.run(feeds);

  mod.HEAPF32.set(output[probsName].data, probsPtr >> 2);
  mod.HEAPF32.set(output[wdlName].data, wdlPtr >> 2);
}

// Above this, a backend is doing something pathological rather than merely being
// slow. A healthy GPU runs this batch in single-digit milliseconds and even the
// wasm CPU backend manages ~220 ms, so anything past half a second is a signal
// to go and measure the alternative.
const SLOW_BACKEND_MS = 500;

function bindNames() {
  inputName = session.inputNames[0];
  // export_az_onnx.py names them; fall back to declaration order for a model
  // from any other exporter.
  probsName = session.outputNames.includes('probs') ? 'probs' : session.outputNames[0];
  wdlName = session.outputNames.includes('wdl') ? 'wdl' : session.outputNames[1];
}

// Runs the network twice at the shape searches actually use, and returns how
// long the SECOND run took.
//
// Two runs, not one, and the second is the one that counts: WebGPU compiles a
// shader for every convolution and gemm the first time it sees a shape, which
// took 11 s here. Timing the first run would measure the compiler, not the
// backend. Doing it at boot also means the compile is paid while the user is
// already waiting for a 40 MB download, instead of inside the first search where
// it ate the entire think budget and left the bot choosing from a 16-evaluation
// tree - which looks like an engine playing a1, b1, rather than a slow one.
//
// The shape must match WebOnnxSession's padded batch or this warms a shape no
// search will ask for.
async function timeWarmup(batch) {
  const feeds = {};
  feeds[inputName] = new ort.Tensor('float32', new Float32Array(batch * OBSERVATION_SIZE), [batch, 4, 8, 8]);
  await session.run(feeds);
  const t0 = performance.now();
  await session.run(feeds);
  return performance.now() - t0;
}

async function boot({ modelUrl, thinkMs, batch, backend }) {
  const forced = backend === 'webgpu' || backend === 'wasm' ? backend : null;
  ort.env.wasm.wasmPaths = ORT_DIST;

  // WebGPU first, wasm as the fallback so the page still works on a browser
  // without it - the same network, just slower.
  //
  // The fallback reason is reported to the UI rather than only to the console.
  // "WebGPU unavailable" has at least three distinct causes - the API is not
  // exposed at all, no adapter can be acquired (the usual one on Linux, where
  // browsers still gate WebGPU behind a flag), or ort failed to build its
  // pipeline - and they need different fixes.
  let detail = '';
  try {
    if (forced === 'wasm') throw new Error('CPU backend selected');
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('navigator.gpu is not exposed - this browser has WebGPU disabled or does not support it');
    }
    // high-performance is not a nicety on hybrid-graphics machines: with no
    // preference the browser hands back whatever it likes, which on a laptop or
    // a desktop with both an integrated and a discrete GPU is usually the
    // integrated one. On an Ivy Bridge box that means Intel HD 4000, which has
    // no working Vulkan driver at all, so WebGPU fails even though a perfectly
    // good discrete card is sitting right there.
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('navigator.gpu.requestAdapter() returned null - no usable GPU adapter');
    }

    // The adapter above is only ours: onnxruntime-web requests its own, and with
    // no preference stated it takes whatever the browser offers first. On a
    // machine with both an integrated and a discrete GPU that is the integrated
    // one, and the difference is not subtle - batch-16 inference measured ~5 s
    // on the wrong adapter against milliseconds on the right one, while still
    // reporting "webgpu" either way. Hand ORT the adapter we already vetted
    // where the build allows it, and state the preference regardless.
    ort.env.webgpu.powerPreference = 'high-performance';
    if ('adapter' in ort.env.webgpu) ort.env.webgpu.adapter = adapter;

    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });
    provider = 'webgpu';
  } catch (err) {
    detail = forced === 'wasm' ? '' : String(err && err.message ? err.message : err);
    if (forced !== 'wasm') console.warn('WebGPU unavailable, falling back to wasm:', err);
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    provider = 'wasm (cpu)';
  }

  bindNames();

  // Pick the backend by measuring it, rather than assuming the GPU wins.
  //
  // On this project's own dev machine WebGPU is available, reports itself
  // healthy, and runs a batch of 16 in ~5000 ms - while the CPU backend does the
  // same work in ~220 ms. A 23x loss, silently, on a page that says "webgpu".
  // Hardware and driver quality vary far too much to hardcode a winner, and the
  // cost of being wrong is a bot that takes minutes per move.
  //
  // The measurement is nearly free: the warm-up run has to happen anyway.
  let ms = await timeWarmup(batch);
  if (!forced && provider === 'webgpu' && ms > SLOW_BACKEND_MS) {
    console.warn(`WebGPU ran a batch in ${ms.toFixed(0)}ms; trying the CPU backend`);
    const gpuMs = ms;
    const gpuSession = session;
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      bindNames();
      const cpuMs = await timeWarmup(batch);
      if (cpuMs < gpuMs) {
        provider = 'wasm (cpu)';
        detail = `CPU ${cpuMs.toFixed(0)}ms/batch beat WebGPU's ${gpuMs.toFixed(0)}ms`;
        ms = cpuMs;
        if (gpuSession.release) await gpuSession.release();
      } else {
        session = gpuSession;      // GPU still wins; keep it
        bindNames();
        ms = gpuMs;
      }
    } catch (e) {
      console.warn('CPU fallback failed, keeping WebGPU:', e);
      session = gpuSession;
      bindNames();
      ms = gpuMs;
    }
  }
  if (!detail) detail = `${ms.toFixed(0)}ms per batch of ${batch}`;

  mod = await createMigoyugoAz({
    locateFile: (p) => new URL(p, self.location.href).href,
    print: (t) => console.log('[az]', t),
    printErr: (t) => console.warn('[az]', t),
  });

  // Installed on both, matching what onnx_session_web.cpp looks for.
  mod.azRun = azRun;
  self.azRun = azRun;

  const rc = mod._mgy_az_init(thinkMs | 0, batch | 0);
  if (rc !== OK) throw new Error(`the engine failed to start: ${describe(rc)}`);

  movePtr = mod._malloc(MAX_PLIES); // scratch for load_moves, allocated once
  snapLen = mod._mgy_az_snapshot_size();
  if (snapLen !== AZ_SNAPSHOT_SIZE) {
    throw new Error(`snapshot size ${snapLen} does not match az_snapshot.js (${AZ_SNAPSHOT_SIZE})`);
  }

  postMessage({ type: 'ready', provider, detail, bridge: BRIDGE, snapshotSize: snapLen });
}

self.onmessage = async (e) => {
  const msg = e.data;
  const epoch = msg.epoch | 0;

  try {
    switch (msg.type) {
      case 'init':
        await boot(msg);
        sendState(epoch);
        break;

      case 'newGame':
        mod._mgy_az_new_game();
        sendState(epoch);
        break;

      case 'play': {
        const rc = mod._mgy_az_play(msg.sq);
        if (rc !== OK) postMessage({ type: 'rejected', epoch, sq: msg.sq, reason: describe(rc) });
        else sendState(epoch);
        break;
      }

      case 'undo': {
        const rc = mod._mgy_az_undo(msg.plies || 1);
        if (rc !== OK) postMessage({ type: 'rejected', epoch, sq: -1, reason: describe(rc) });
        else sendState(epoch);
        break;
      }

      case 'botMove': {
        postMessage({ type: 'thinking', epoch });
        const t0 = performance.now();
        // ccall with async:true, NOT mod._mgy_az_bot_move(). The search suspends
        // every time it hands a batch to the GPU, and a directly-called Asyncify
        // export returns a meaningless value at the first suspend rather than
        // waiting for the function to finish.
        //
        // The same form is correct under JSPI (emscripten src/lib/libccall.js
        // returns ret.then(onDone) there), so no branch is needed - but note the
        // asymmetry: under JSPI, omitting {async: true} fails SILENTLY, handing
        // back a converted Promise object with no assertion to catch it.
        const sq = await mod.ccall('mgy_az_bot_move', 'number', [], [], { async: true });
        const elapsed = performance.now() - t0;
        if (sq < 0) {
          postMessage({ type: 'rejected', epoch, sq: -1, reason: describe(sq) });
        } else {
          sendState(epoch, {
            botMove: sq,
            elapsedMs: elapsed,
            evaluation: mod._mgy_az_evaluation(),
            probs: probsArray(),
          });
        }
        break;
      }

      // The message index.html uses. This module is NOT the authoritative
      // position there - the NNUE module is - so the position arrives as a move
      // list, is replayed, searched, and the answer handed back without ever
      // being applied here. The caller plays it through the NNUE module so the
      // rich snapshot that drives the board keeps coming from one place.
      case 'suggest': {
        const moves = msg.moves || [];
        if (moves.length > MAX_PLIES) {
          postMessage({ type: 'error', epoch, message: `move list too long (${moves.length})` });
          break;
        }
        mod.HEAPU8.set(new Uint8Array(moves), movePtr); // fresh view, never cached
        const rc = mod._mgy_az_load_moves(movePtr, moves.length);
        if (rc !== OK) {
          // A rejection here means MigoyugoBB and MigoyugoState disagree about
          // legality. That is a real bug, not a user error - say so loudly.
          postMessage({ type: 'error', epoch, message: `position resync failed: ${describe(rc)}` });
          break;
        }

        postMessage({ type: 'thinking', epoch });
        const t0 = performance.now();
        const sq = await mod.ccall('mgy_az_bot_suggest', 'number', [], [], { async: true });
        const elapsed = performance.now() - t0;

        if (sq < 0) {
          postMessage({ type: 'rejected', epoch, sq: -1, reason: describe(sq) });
        } else {
          searchSucceeded = true;
          postMessage({
            type: 'azMove', epoch, sq,
            elapsedMs: elapsed,
            evaluation: mod._mgy_az_evaluation(),
            evaluations: mod._mgy_az_last_evaluations(),
            probs: probsArray(),
          });
        }
        break;
      }

      case 'setTime':
        mod._mgy_az_set_time_ms(msg.ms | 0);
        break;

      case 'setBatch':
        mod._mgy_az_set_batch(msg.batch | 0);
        break;

      default:
        console.warn('az_worker: unknown message', msg.type);
    }
  } catch (err) {
    const message = String(err && err.message ? err.message : err);

    // Feature detection can pass on a partial or origin-trial JSPI while the
    // build still fails - at module instantiation, or at the first real suspend.
    // Neither is recoverable inside this worker: the glue was chosen by
    // importScripts at load time and cannot be swapped. Hand it back to app.js,
    // which respawns with ?bridge=asyncify.
    if (BRIDGE === 'jspi' && !searchSucceeded) {
      console.warn('JSPI bridge failed, falling back to asyncify:', err);
      postMessage({ type: 'bridgeFailed', epoch, bridge: BRIDGE, message });
      return;
    }
    postMessage({ type: 'error', epoch, message });
  }
};
