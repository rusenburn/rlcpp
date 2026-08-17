// Controller: owns the UI and the game mode, and talks to the worker.
//
// It holds no rules. Whether a move is legal, what it captures and whether the
// game is over are all answered by the C++ engine and arrive in the snapshot.
// The move list kept here is only so a crashed or terminated worker can be
// respawned and replayed.
//
// Every request carries an epoch, bumped on new-game / undo / stop / mode
// change. The worker echoes it and replies with a stale epoch are dropped,
// which is what makes stopping mid-search safe: the in-flight reply is
// discarded and the next move is simply never scheduled.

import {
  parseSnapshot, parseInfo, SNAPSHOT_SIZE, INFO_SIZE,
  PLAYING, IGO, WEGO, WHITE, BLACK, squareName, formatScore, toWhiteScore, isWhite,
} from './snapshot.js';
import { Board } from './board.js';
import { Sound } from './sound.js';

const $ = (id) => document.getElementById(id);

// Who owns a colour. HUMAN/NNUE/AZ per side replaces the old mode enum: every
// arrangement it could express is a pair of these, plus the ones it could not
// (NNUE vs AlphaYugo, and either engine against itself).
const HUMAN = 'human';
const NNUE = 'nnue';
const AZ = 'az';

const ENGINE_LABEL = { [NNUE]: 'NNUE', [AZ]: 'AlphaYugo' };

// Leaves handed to the network per WebGPU call (Amcts2's max_async_simulations).
//
// Chosen for move quality, not throughput - and the two disagree here. Measured
// on a 3060 Ti: batch 16 runs ~16 ms and yields 830-1000 evaluations/s, batch 8
// runs ~30 ms and yields 375-480, because 8 positions do not saturate the GPU
// and pay almost the same per-call overhead. The case for 8 is that a smaller
// batch re-expands the tree twice as often, so fewer leaves are selected against
// a stale tree; that can win on strength while losing on raw count. If it is
// ever settled by a head-to-head match, record the result here.
//
// Changing this rebuilds the ONNX session: WebOnnxSession pads every run to one
// fixed shape, and WebGPU compiles shaders per shape.
const AZ_BATCH = 8;

const state = {
  engines: { [WHITE]: HUMAN, [BLACK]: NNUE },
  epoch: 0,
  snapshot: null,
  info: null,
  // Which colour the engine was searching for when `info` was produced. The
  // score is relative to that colour, and it is NOT the snapshot's stm once
  // the move has been played.
  infoStm: null,
  // Which engine produced `info`, so the panel can label its units. NNUE has a
  // depth and counts nodes; AlphaYugo has neither.
  infoEngine: null,
  moves: [],
  thinking: false,
  ready: false,     // NNUE worker
  azReady: false,   // AlphaYugo worker, many seconds later - 40 MB plus shaders
  azLoading: false,
  running: false,   // engine-vs-engine loop is live
  hint: -1,
};

const sound = new Sound();
let board = null;
let worker = null;
let azWorker = null;

// --- worker plumbing --------------------------------------------------------

function spawnWorker() {
  if (worker) worker.terminate();
  worker = new Worker('worker.js');
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => fatal(e.message || 'the engine worker failed to start');
  worker.postMessage({
    type: 'init',
    modelUrl: 'migoyugo_nnue_v2.bin',
    ttMb: 16,
  });
}

// Spawned on demand, the first time a side is set to AlphaYugo. Its network is
// 40 MB and its WebGPU shaders take seconds to compile, so anyone who only ever
// plays the NNUE engine or another human must not pay for it.
function spawnAzWorker() {
  if (azWorker || state.azLoading) return;
  state.azLoading = true;
  setAzStatus('AlphaYugo: loading the network (40 MB)…');

  azWorker = new Worker('az_worker.js');
  azWorker.onmessage = onAzMessage;
  azWorker.onerror = (e) => {
    state.azLoading = false;
    setAzStatus(`AlphaYugo failed to start: ${e.message || 'worker error'}`, true);
  };
  azWorker.postMessage({
    type: 'init',
    epoch: state.epoch,
    modelUrl: 'migoyugo_az.onnx',
    thinkMs: Number($('think').value),
    batch: AZ_BATCH,
    backend: $('az-backend').value,
  });
}

function send(type, payload = {}) {
  worker.postMessage({ type, epoch: state.epoch, ...payload });
}

function azSend(type, payload = {}) {
  azWorker.postMessage({ type, epoch: state.epoch, ...payload });
}

function setAzStatus(text, isError = false) {
  const el = $('az-status');
  el.hidden = !text;
  el.textContent = text;
  el.classList.toggle('error', isError);
}

function bumpEpoch() {
  state.epoch++;
  state.hint = -1;
}

function onWorkerMessage(e) {
  const msg = e.data;

  if (msg.type === 'ready') {
    if (msg.snapshotSize !== SNAPSHOT_SIZE || msg.infoSize !== INFO_SIZE) {
      fatal(`engine/page layout mismatch: the module reports ${msg.snapshotSize}/${msg.infoSize} ` +
            `bytes, this page expects ${SNAPSHOT_SIZE}/${INFO_SIZE}. Hard-refresh to clear the cache.`);
      return;
    }
    state.ready = true;
    setStatus('Ready.');
    newGame();
    return;
  }

  if (msg.type === 'error') {
    fatal(msg.message);
    return;
  }

  // Anything below is a reply to a specific request; drop it if the world has
  // moved on since.
  if (msg.epoch !== state.epoch) return;

  switch (msg.type) {
    case 'thinking':
      state.thinking = true;
      render();
      break;

    case 'rejected':
      state.thinking = false;
      if (msg.sq >= 0) board.reject(msg.sq);
      sound.illegal();
      setStatus(msg.reason);
      render();
      break;

    case 'hint':
      state.thinking = false;
      state.hint = msg.sq;
      state.info = parseInfo(msg.info);
      state.infoStm = state.snapshot ? state.snapshot.stm : null;
      setStatus(`Suggestion: ${squareName(msg.sq)}`);
      render();
      break;

    case 'state': {
      state.thinking = false;
      const previous = state.snapshot;
      state.snapshot = parseSnapshot(msg.snapshot);
      // A move found without searching (an instant Igo, or the only legal
      // move) reports no depth or nodes. Keep the last real search on screen
      // rather than blanking the panel exactly when the game ends.
      const info = parseInfo(msg.info);
      if (msg.botMove !== undefined && (info.nodes > 0 || info.depth > 0)) {
        state.info = info;
        state.infoEngine = NNUE;
        // The search ran before the move, so it scored for the mover, who is
        // no longer the side to move.
        state.infoStm = 1 - state.snapshot.stm;
      }
      state.hint = -1;
      syncMoves(previous);
      playSounds(previous);
      render();
      scheduleNext();
      break;
    }
  }
}

// AlphaYugo replies. Deliberately a separate handler from onWorkerMessage: this
// worker is not the authoritative position, only a move oracle, so its `state`
// messages (which az.html uses) are ignored here.
//
// Its failures are also not fatal() material - the NNUE engine and the board are
// still perfectly usable if the GPU path dies, so this reports and stops rather
// than throwing up the overlay.
function onAzMessage(e) {
  const msg = e.data;

  if (msg.type === 'ready') {
    state.azLoading = false;
    state.azReady = true;
    setAzStatus(`AlphaYugo: ready on ${msg.provider}.`);
    azWorker.postMessage({ type: 'setTime', ms: Number($('think').value) });
    render();
    scheduleNext();
    return;
  }

  if (msg.type === 'error') {
    state.azLoading = false;
    state.thinking = false;
    state.running = false;
    setAzStatus(`AlphaYugo: ${msg.message}`, true);
    render();
    return;
  }

  if (msg.epoch !== state.epoch) return;

  switch (msg.type) {
    case 'thinking':
      state.thinking = true;
      render();
      break;

    case 'rejected':
      state.thinking = false;
      state.running = false;
      setAzStatus(`AlphaYugo: ${msg.reason}`, true);
      render();
      break;

    case 'azMove':
      // Record what it thought, then play the move through the NNUE module.
      // That module owns the position of record and is the only source of a
      // snapshot rich enough for the board - captures, win lines, forbidden
      // squares - so every move, whoever chose it, lands the same way.
      state.info = {
        evaluation: msg.evaluation,
        evaluations: msg.evaluations,
        elapsedMs: msg.elapsedMs,
      };
      state.infoEngine = AZ;
      state.infoStm = state.snapshot ? state.snapshot.stm : null;
      send('play', { sq: msg.sq });
      break;
  }
}

function fatal(message) {
  state.ready = false;
  state.running = false;
  setStatus(message, true);
  $('overlay').hidden = false;
  $('overlay-title').textContent = 'Something went wrong';
  $('overlay-text').textContent = message;
  $('overlay-action').hidden = false;
}

// --- move bookkeeping (for worker recovery only) ---------------------------

function syncMoves(previous) {
  const s = state.snapshot;
  if (s.moveCount === 0) { state.moves = []; return; }
  if (previous && s.moveCount === previous.moveCount + 1 && s.lastMove >= 0) {
    state.moves.push(s.lastMove);
  } else if (s.moveCount < state.moves.length) {
    state.moves.length = s.moveCount; // an undo
  }
  // If the counts still disagree the move list is only used for recovery, so a
  // stale entry is harmless; it is rebuilt on the next new game.
}

// --- sound ------------------------------------------------------------------

function playSounds(previous) {
  const s = state.snapshot;
  if (!previous || s.moveCount === 0) return;
  if (s.moveCount <= previous.moveCount) return; // undo, or a reload

  if (s.lastPromotion) sound.promote();
  else sound.place(s.lastMove >= 0 && isWhite(s.cells[s.lastMove]));

  if (s.cleared.length) sound.capture(s.cleared.length);

  if (s.status !== PLAYING) {
    if (s.winner === null) sound.draw();
    else if (soloHuman()) {
      s.winner === humanSeat() ? sound.win() : sound.lose();
    } else sound.win();
  }
}

// --- turn logic -------------------------------------------------------------

function engineFor(seat) { return state.engines[seat]; }
function seatIsBot(seat) { return engineFor(seat) !== HUMAN; }

// True when a human is sitting at exactly one side - the only arrangement where
// "you won" / "you lost" and taking back a pair of plies make sense.
function soloHuman() {
  const w = engineFor(WHITE) === HUMAN;
  const b = engineFor(BLACK) === HUMAN;
  return w !== b;
}
function humanSeat() { return engineFor(WHITE) === HUMAN ? WHITE : BLACK; }
function bothEngines() { return !(engineFor(WHITE) === HUMAN || engineFor(BLACK) === HUMAN); }

function scheduleNext() {
  const s = state.snapshot;
  if (!s || s.status !== PLAYING) {
    state.running = false;
    render();
    return;
  }
  // Engine vs engine only advances while the Start button says so; with a human
  // on one side, the engine always answers.
  if (bothEngines() && !state.running) return;

  const engine = engineFor(s.stm);
  if (engine === HUMAN) return;

  if (engine === AZ) {
    // Still fetching the 40 MB network or compiling shaders. scheduleNext is
    // called again from the ready handler.
    if (!state.azReady) { spawnAzWorker(); return; }
    // The oracle keeps no position of its own here: hand it the move list and
    // let it resync before it searches.
    azSend('suggest', { moves: state.moves.slice() });
    return;
  }
  send('botMove');
}

// --- actions ----------------------------------------------------------------

function newGame() {
  bumpEpoch();
  state.moves = [];
  state.snapshot = null;
  state.info = null;
  state.infoStm = null;
  state.infoEngine = null;
  state.running = bothEngines() ? state.running : false;
  send('newGame');
}

function play(sq) {
  sound.resume();
  const s = state.snapshot;
  if (!s || s.status !== PLAYING || state.thinking) return;
  if (seatIsBot(s.stm)) return;
  send('play', { sq });
}

function undo() {
  const s = state.snapshot;
  if (!s || !s.canUndo) return;
  // With a human on one side only, take back the pair so it is their turn again.
  const plies = soloHuman() && s.moveCount >= 2 && seatIsBot(1 - s.stm) ? 2 : 1;
  bumpEpoch();
  state.running = false;
  send('undo', { plies });
}

function stop() {
  bumpEpoch();
  state.running = false;
  state.thinking = false;
  setStatus('Stopped.');
  render();
}

function hint() {
  const s = state.snapshot;
  if (!s || s.status !== PLAYING || state.thinking) return;
  sound.resume();
  send('hint');
}

// --- rendering --------------------------------------------------------------

function setStatus(text, isError = false) {
  const el = $('status');
  el.textContent = text;
  el.classList.toggle('error', isError);
}

function describeEnding(s) {
  if (s.status === IGO) {
    const who = s.winner === WHITE ? 'White' : 'Black';
    return `${who} wins by Igo — four Yugos in a line.`;
  }
  const [w, b] = s.yugoCount;
  if (s.winner === null) return `Drawn by Wego — ${w} Yugos each.`;
  const who = s.winner === WHITE ? 'White' : 'Black';
  return `${who} wins by Wego — no legal move left, ${w}–${b} on Yugos.`;
}

function render() {
  const s = state.snapshot;
  if (!s) return;

  const humanToMove = s.status === PLAYING && !seatIsBot(s.stm) && !state.thinking;
  board.setInteractive(humanToMove);
  board.render(s, {
    animate: true,
    showPiline: $('opt-piline').checked && s.status === PLAYING,
    showHints: $('opt-hints').checked,
    hint: state.hint,
  });

  $('score-w').textContent = String(s.yugoCount[WHITE]);
  $('score-b').textContent = String(s.yugoCount[BLACK]);
  $('turn-w').classList.toggle('active', s.status === PLAYING && s.stm === WHITE);
  $('turn-b').classList.toggle('active', s.status === PLAYING && s.stm === BLACK);

  renderEngineInfo();

  $('btn-undo').disabled = !s.canUndo || state.thinking;
  $('btn-hint').disabled = s.status !== PLAYING || state.thinking;
  $('btn-stop').hidden = !(state.thinking || state.running);
  $('btn-run').hidden = !bothEngines() || state.running || s.status !== PLAYING;

  renderMoveList(s);

  if (s.status !== PLAYING) {
    setStatus(`Game over after ${s.moveCount} moves.`);
    $('banner').textContent = describeEnding(s);
    $('banner').hidden = false;
  } else {
    $('banner').hidden = true;
    if (state.thinking) setStatus('Thinking…');
    else if (seatIsBot(s.stm)) setStatus(`${ENGINE_LABEL[engineFor(s.stm)]} to move.`);
    else setStatus(`${s.stm === WHITE ? 'White' : 'Black'} to move.`);
  }
}

// The two engines have nothing in common numerically: NNUE searches to a depth
// and counts alpha-beta nodes in engine units of 1/1024; AlphaYugo has no depth
// at all, scores in [-1, 1], and counts network evaluations. So the labels move
// with the engine rather than the numbers being forced into one shape.
function renderEngineInfo() {
  const i = state.info;
  const engine = state.infoEngine;
  const blank = () => {
    $('eng-name').textContent = '';
    for (const id of ['eng-depth', 'eng-score', 'eng-nodes', 'eng-nps']) $(id).textContent = '—';
  };

  if (!i || state.infoStm === null || !engine) { blank(); return; }
  $('eng-name').textContent = ENGINE_LABEL[engine];

  if (engine === AZ) {
    $('lbl-depth').textContent = 'Depth';
    $('lbl-nodes').textContent = 'Evals';
    $('lbl-speed').textContent = 'Speed';
    // Depth is meaningless for a Monte-Carlo search; say so rather than invent one.
    $('eng-depth').textContent = 'n/a';
    // Same sign convention as the NNUE score: positive favours White.
    const white = toWhiteScore(i.evaluation, state.infoStm);
    $('eng-score').textContent = `${white >= 0 ? '+' : ''}${white.toFixed(3)}`;
    $('eng-nodes').textContent = i.evaluations.toLocaleString();
    const secs = i.elapsedMs / 1000;
    $('eng-nps').textContent = secs > 0 ? `${Math.round(i.evaluations / secs).toLocaleString()} ev/s` : '—';
    return;
  }

  // A move found without searching - an instant Igo, or the only legal move -
  // reports no depth and no nodes.
  if (!(i.depth > 0)) { blank(); return; }
  $('lbl-depth').textContent = 'Depth';
  $('lbl-nodes').textContent = 'Nodes';
  $('lbl-speed').textContent = 'Speed';
  $('eng-depth').textContent = String(i.depth);
  $('eng-score').textContent = formatScore(toWhiteScore(i.score, state.infoStm));
  $('eng-nodes').textContent = Math.round(i.nodes).toLocaleString();
  $('eng-nps').textContent = `${Math.round(i.nps / 1000).toLocaleString()}k n/s`;
}

function renderMoveList(s) {
  const list = $('moves');
  const wanted = state.moves.length;
  if (list.childElementCount === wanted && wanted > 0) {
    // Only the highlight can have changed.
    for (const el of list.children) el.classList.toggle('current', Number(el.dataset.i) === wanted - 1);
    return;
  }
  list.innerHTML = '';
  state.moves.forEach((sq, i) => {
    const el = document.createElement('li');
    el.dataset.i = String(i);
    el.textContent = squareName(sq);
    if (i === wanted - 1) el.classList.add('current');
    list.appendChild(el);
  });
  list.scrollTop = list.scrollHeight;
  void s;
}

// --- wiring -----------------------------------------------------------------

function applySeats() {
  state.engines[WHITE] = $('engine-w').value;
  state.engines[BLACK] = $('engine-b').value;

  // The think-time row is pointless with no engine on the board.
  $('bot-row').hidden = !seatIsBot(WHITE) && !seatIsBot(BLACK);

  // Orient the board for the only human, if there is exactly one.
  if (soloHuman()) {
    const flip = humanSeat() === BLACK;
    board.setFlipped(flip);
    $('opt-flip').checked = flip;
  }

  // Start fetching the network as soon as a side is set to AlphaYugo, rather
  // than at the moment it is first asked to move - the 40 MB and the shader
  // compile would otherwise land as a stall mid-game.
  const azInPlay = engineFor(WHITE) === AZ || engineFor(BLACK) === AZ;
  $('az-backend-row').hidden = !azInPlay;
  if (azInPlay) spawnAzWorker();

  bumpEpoch();
  state.running = false;
  if (state.snapshot) { render(); scheduleNext(); }
}

function init() {
  board = new Board($('board'), play);

  $('engine-w').addEventListener('change', () => { applySeats(); newGame(); });
  $('engine-b').addEventListener('change', () => { applySeats(); newGame(); });

  $('az-backend').addEventListener('change', () => {
    // The backend is fixed when the ONNX session is created, so switching means
    // a fresh worker: a new session, a new warm-up, another 40 MB parse.
    if (azWorker) { azWorker.terminate(); azWorker = null; }
    state.azReady = false;
    state.azLoading = false;
    state.running = false;
    bumpEpoch();
    spawnAzWorker();
  });

  $('think').addEventListener('input', () => {
    const ms = Number($('think').value);
    $('think-label').textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`;
    // Both engines share the clock, so an engine-vs-engine game is a fair test.
    worker.postMessage({ type: 'setTime', ms });
    if (azWorker && state.azReady) azWorker.postMessage({ type: 'setTime', ms });
  });

  $('btn-new').addEventListener('click', () => { sound.resume(); newGame(); });
  $('btn-undo').addEventListener('click', () => { sound.resume(); undo(); });
  $('btn-hint').addEventListener('click', hint);
  $('btn-stop').addEventListener('click', stop);
  $('btn-run').addEventListener('click', () => {
    sound.resume();
    state.running = true;
    render();
    scheduleNext();
  });
  $('overlay-action').addEventListener('click', () => location.reload());

  $('opt-flip').addEventListener('change', (e) => board.setFlipped(e.target.checked));
  $('opt-piline').addEventListener('change', render);
  $('opt-hints').addEventListener('change', render);
  $('opt-sound').addEventListener('change', (e) => sound.setEnabled(e.target.checked));

  $('theme').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme !== 'light';
    document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  });

  // A backgrounded tab throttles the message loop, so a bot-vs-bot game would
  // appear to stall. Pause it deliberately instead.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.running) stop();
  });

  applySeats();
  const ms = Number($('think').value);
  $('think-label').textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`;

  setStatus('Loading engine…');
  spawnWorker();
  worker.addEventListener('message', function once(e) {
    if (e.data.type === 'ready') {
      worker.postMessage({ type: 'setTime', ms: Number($('think').value) });
      worker.removeEventListener('message', once);
    }
  });
}

init();
