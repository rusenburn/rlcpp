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

const MODE_HUMAN_BOT = 'human-bot';
const MODE_BOT_BOT = 'bot-bot';
const MODE_HUMAN_HUMAN = 'human-human';

const state = {
  mode: MODE_HUMAN_BOT,
  humanSeat: WHITE,
  epoch: 0,
  snapshot: null,
  info: null,
  // Which colour the engine was searching for when `info` was produced. The
  // score is relative to that colour, and it is NOT the snapshot's stm once
  // the move has been played.
  infoStm: null,
  moves: [],
  thinking: false,
  ready: false,
  running: false, // bot-vs-bot loop is live
  hint: -1,
};

const sound = new Sound();
let board = null;
let worker = null;

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

function send(type, payload = {}) {
  worker.postMessage({ type, epoch: state.epoch, ...payload });
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
    else if (state.mode === MODE_HUMAN_BOT) {
      s.winner === state.humanSeat ? sound.win() : sound.lose();
    } else sound.win();
  }
}

// --- turn logic -------------------------------------------------------------

function seatIsBot(seat) {
  if (state.mode === MODE_HUMAN_HUMAN) return false;
  if (state.mode === MODE_BOT_BOT) return true;
  return seat !== state.humanSeat;
}

function scheduleNext() {
  const s = state.snapshot;
  if (!s || s.status !== PLAYING) {
    state.running = false;
    render();
    return;
  }
  if (state.mode === MODE_BOT_BOT && !state.running) return;
  if (!seatIsBot(s.stm)) return;
  send('botMove');
}

// --- actions ----------------------------------------------------------------

function newGame() {
  bumpEpoch();
  state.moves = [];
  state.snapshot = null;
  state.info = null;
  state.infoStm = null;
  state.running = state.mode === MODE_BOT_BOT ? state.running : false;
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
  // In Human vs Bot, take back the pair so it is the human's turn again.
  const plies = state.mode === MODE_HUMAN_BOT && s.moveCount >= 2 && seatIsBot(1 - s.stm) ? 2 : 1;
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

  const i = state.info;
  const hasInfo = i && i.depth > 0 && state.infoStm !== null;
  $('eng-depth').textContent = hasInfo ? String(i.depth) : '—';
  $('eng-score').textContent = hasInfo ? formatScore(toWhiteScore(i.score, state.infoStm)) : '—';
  $('eng-nodes').textContent = hasInfo ? Math.round(i.nodes).toLocaleString() : '—';
  $('eng-nps').textContent = hasInfo ? `${Math.round(i.nps / 1000).toLocaleString()}k` : '—';

  $('btn-undo').disabled = !s.canUndo || state.thinking;
  $('btn-hint').disabled = s.status !== PLAYING || state.thinking;
  $('btn-stop').hidden = !(state.thinking || state.running);
  $('btn-run').hidden = state.mode !== MODE_BOT_BOT || state.running || s.status !== PLAYING;

  renderMoveList(s);

  if (s.status !== PLAYING) {
    setStatus(`Game over after ${s.moveCount} moves.`);
    $('banner').textContent = describeEnding(s);
    $('banner').hidden = false;
  } else {
    $('banner').hidden = true;
    if (state.thinking) setStatus('Thinking…');
    else if (seatIsBot(s.stm)) setStatus('Engine to move.');
    else setStatus(`${s.stm === WHITE ? 'White' : 'Black'} to move.`);
  }
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

function applyMode() {
  state.mode = $('mode').value;
  $('seat-row').hidden = state.mode !== MODE_HUMAN_BOT;
  $('bot-row').hidden = state.mode === MODE_HUMAN_HUMAN;
  bumpEpoch();
  state.running = false;
  if (state.snapshot) { render(); scheduleNext(); }
}

function init() {
  board = new Board($('board'), play);

  $('mode').addEventListener('change', () => { applyMode(); newGame(); });
  $('seat').addEventListener('change', () => {
    state.humanSeat = $('seat').value === 'black' ? BLACK : WHITE;
    board.setFlipped(state.humanSeat === BLACK);
    $('opt-flip').checked = state.humanSeat === BLACK;
    newGame();
  });

  $('think').addEventListener('input', () => {
    const ms = Number($('think').value);
    $('think-label').textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`;
    worker.postMessage({ type: 'setTime', ms });
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

  applyMode();
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
