// Owns the WebAssembly module and the authoritative game position. The page
// is a pure view: it never decides whether a move is legal, what a move
// captures, or whether the game is over - it asks, and renders the answer.
//
// A classic worker on purpose: importScripts with -sMODULARIZE is the
// best-supported emscripten path, and this file needs none of the ES-module
// machinery because it does no parsing - it slices bytes and forwards them.
//
// The one rule that matters here: cache the wasm POINTER, never the heap VIEW.
// -sALLOW_MEMORY_GROWTH replaces HEAPU8 on every grow and detaches the old
// one, so a cached view is a time bomb that only goes off once someone picks a
// bigger transposition table. Pointers into linear memory stay valid.

'use strict';

importScripts('migoyugo.js'); // defines self.createMigoyugo

let mod = null;
let snapLen = 0, infoLen = 0;
let movePtr = 0; // scratch for load_moves, allocated once

const OK = 0;

const ERRORS = {
  '-1': 'no model loaded',
  '-2': 'square out of range',
  '-3': 'illegal move',
  '-4': 'the game is already over',
  '-5': 'the weights file could not be read',
  '-6': 'model memory was not correctly aligned',
  '-7': 'nothing to undo',
};
const describe = (code) => ERRORS[String(code)] || `error ${code}`;

// Copy out of linear memory into a standalone buffer so the main thread never
// holds a window into the wasm heap. slice(), not subarray().
//
// _mgy_snapshot() is NOT a pure getter: it rewrites the struct from the
// current position and returns its address. Calling it once and then reading
// the remembered address forever leaves the board frozen at the initial
// position, so it is called on every read. The address itself never moves;
// re-reading mod.HEAPU8 each time is what guards against memory growth having
// detached the previous view.
function snapshotBuffer() {
  const p = mod._mgy_snapshot();
  return mod.HEAPU8.slice(p, p + snapLen).buffer;
}
function infoBuffer() {
  const p = mod._mgy_info();
  return mod.HEAPU8.slice(p, p + infoLen).buffer;
}

function sendState(epoch, extra = {}) {
  const snapshot = snapshotBuffer();
  const info = infoBuffer();
  postMessage({ type: 'state', epoch, snapshot, info, ...extra }, [snapshot, info]);
}

async function boot({ modelUrl, ttMb }) {
  mod = await createMigoyugo({
    locateFile: (p) => new URL(p, self.location.href).href,
    print: (t) => console.log('[wasm]', t),
    printErr: (t) => console.warn('[wasm]', t),
  });

  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error(`${modelUrl}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());

  const p = mod._malloc(bytes.length);
  if (!p) throw new Error('out of memory loading the network');
  mod.HEAPU8.set(bytes, p); // fresh view, read after the malloc that may have grown the heap
  const rc = mod._mgy_init(p, bytes.length, ttMb);
  mod._free(p);
  if (rc !== OK) throw new Error(`the engine rejected the weights: ${describe(rc)}`);

  snapLen = mod._mgy_snapshot_size();
  infoLen = mod._mgy_info_size();
  movePtr = mod._malloc(64);

  postMessage({ type: 'ready', snapshotSize: snapLen, infoSize: infoLen });
}

self.onmessage = async (e) => {
  const msg = e.data;
  const epoch = msg.epoch | 0;

  try {
    switch (msg.type) {
      case 'init':
        await boot(msg);
        break;

      case 'newGame':
        mod._mgy_new_game();
        sendState(epoch);
        break;

      case 'play': {
        const rc = mod._mgy_play(msg.sq);
        if (rc !== OK) postMessage({ type: 'rejected', epoch, sq: msg.sq, reason: describe(rc) });
        else sendState(epoch);
        break;
      }

      case 'undo': {
        const rc = mod._mgy_undo(msg.plies || 1);
        if (rc !== OK) postMessage({ type: 'rejected', epoch, sq: -1, reason: describe(rc) });
        else sendState(epoch);
        break;
      }

      case 'loadMoves': {
        const moves = msg.moves || [];
        mod.HEAPU8.set(new Uint8Array(moves), movePtr);
        const rc = mod._mgy_load_moves(movePtr, moves.length);
        if (rc !== OK) postMessage({ type: 'error', epoch, message: describe(rc) });
        else sendState(epoch);
        break;
      }

      case 'botMove': {
        postMessage({ type: 'thinking', epoch });
        const sq = mod._mgy_bot_move();
        if (sq < 0) postMessage({ type: 'rejected', epoch, sq: -1, reason: describe(sq) });
        else sendState(epoch, { botMove: sq });
        break;
      }

      case 'hint': {
        postMessage({ type: 'thinking', epoch });
        const sq = mod._mgy_bot_suggest();
        const info = infoBuffer();
        if (sq < 0) postMessage({ type: 'rejected', epoch, sq: -1, reason: describe(sq) });
        else postMessage({ type: 'hint', epoch, sq, info }, [info]);
        break;
      }

      case 'setTime':
        mod._mgy_set_time_ms(msg.ms);
        break;

      case 'setTT':
        mod._mgy_set_tt_mb(msg.mb);
        break;

      default:
        console.warn('worker: unknown message', msg.type);
    }
  } catch (err) {
    postMessage({ type: 'error', epoch, message: String(err && err.message ? err.message : err) });
  }
};
