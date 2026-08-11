// The binary contract with the Snapshot and Info structs in
// wasm/migoyugo_wasm.cpp.
//
// This layout is split across a C++ header and this file, and browsers cache
// aggressively, so a field added on one side and not the other would misparse
// silently. Three cheap checks prevent that: the C++ side has
// static_assert(sizeof(Snapshot) == 416), the module reports its own
// mgy_snapshot_size() in the ready message for app.js to assert against
// SNAPSHOT_SIZE, and parseSnapshot checks the length and version byte here.

export const SNAPSHOT_SIZE = 416;
export const SNAPSHOT_VERSION = 1;
export const INFO_SIZE = 32;

const O = {
  version: 0, stm: 1, status: 2, winner: 3,
  yugo: 4, migo: 6, moveCount: 8, lastMove: 9,
  lastPromotion: 10, nCleared: 11, legalCount: 12, canUndo: 13, flags: 14,
  cleared: 16, winLine: 28,
  cells: 32, legal: 96, promoting: 160, winning: 224, pilineW: 288, pilineB: 352,
};

// Cell codes, matching Game::write_snapshot.
export const EMPTY = 0, W_MIGO = 1, W_YUGO = 2, B_MIGO = 3, B_YUGO = 4;

// status
export const PLAYING = 0, IGO = 1, WEGO = 2;

export const WHITE = 0, BLACK = 1;
export const NO_SQUARE = 255;

export const isWhite = (cell) => cell === W_MIGO || cell === W_YUGO;
export const isYugo = (cell) => cell === W_YUGO || cell === B_YUGO;
export const colorOf = (cell) => (cell === EMPTY ? -1 : isWhite(cell) ? WHITE : BLACK);

/**
 * @param {ArrayBuffer} buf a transferred standalone copy of the C++ Snapshot.
 *   It must NOT be a view into wasm memory - worker.js slices before sending,
 *   which is what makes the subarrays below safe to hold onto.
 */
export function parseSnapshot(buf) {
  const b = new Uint8Array(buf);
  if (b.length !== SNAPSHOT_SIZE) {
    throw new Error(`snapshot is ${b.length} bytes, expected ${SNAPSHOT_SIZE}`);
  }
  if (b[O.version] !== SNAPSHOT_VERSION) {
    throw new Error(`snapshot version ${b[O.version]}, expected ${SNAPSHOT_VERSION}`);
  }

  const n = b[O.nCleared];
  return {
    stm: b[O.stm],
    status: b[O.status],
    winner: b[O.winner] === NO_SQUARE ? null : b[O.winner],
    yugoCount: [b[O.yugo], b[O.yugo + 1]],
    migoCount: [b[O.migo], b[O.migo + 1]],
    moveCount: b[O.moveCount],
    lastMove: b[O.lastMove] === NO_SQUARE ? -1 : b[O.lastMove],
    lastPromotion: b[O.lastPromotion] === 1,
    byBot: (b[O.flags] & 1) !== 0,
    canUndo: b[O.canUndo] === 1,
    legalCount: b[O.legalCount],
    // Plain arrays: these outlive the message, being read during animations.
    cleared: Array.from(b.subarray(O.cleared, O.cleared + n)),
    winLine: Array.from(b.subarray(O.winLine, O.winLine + 4)).filter((s) => s !== NO_SQUARE),
    cells: b.subarray(O.cells, O.cells + 64),
    legal: b.subarray(O.legal, O.legal + 64),
    promoting: b.subarray(O.promoting, O.promoting + 64),
    winning: b.subarray(O.winning, O.winning + 64),
    piline: [
      b.subarray(O.pilineW, O.pilineW + 64),
      b.subarray(O.pilineB, O.pilineB + 64),
    ],
  };
}

/** @param {ArrayBuffer} buf a transferred copy of the C++ Info struct. */
export function parseInfo(buf) {
  if (buf.byteLength !== INFO_SIZE) {
    throw new Error(`info is ${buf.byteLength} bytes, expected ${INFO_SIZE}`);
  }
  const dv = new DataView(buf); // wasm is always little-endian
  return {
    nodes: dv.getFloat64(0, true),
    nps: dv.getFloat64(8, true),
    depth: dv.getInt32(16, true),
    score: dv.getInt32(20, true), // engine units; /1024 for the net's scale
    bestMove: dv.getInt32(24, true),
    elapsedMs: dv.getInt32(28, true),
  };
}

// Square 0 is the top-left of the board, matching MigoyugoBB's bit index
// (row * 8 + col). Files are a-h left to right, ranks 8-1 top to bottom.
export const rowOf = (sq) => (sq >> 3);
export const colOf = (sq) => (sq & 7);
export const squareName = (sq) => `${String.fromCharCode(97 + colOf(sq))}${8 - rowOf(sq)}`;

// Scores at or beyond this are proven wins/losses rather than evaluations.
const MATE = 30000;
const MATE_IN_MAX = MATE - 2 * 96;

/**
 * The engine always scores from the point of view of the side to move AT THE
 * TIME OF THE SEARCH. That is not the same as the side to move in the snapshot
 * that follows it: once the engine plays its move, the turn has flipped. Using
 * the post-move stm reads every evaluation backwards, so the caller must pass
 * the colour the search actually ran for.
 *
 * @param {number} score engine units, positive = good for searchStm
 * @param {number} searchStm WHITE or BLACK - who was to move when this ran
 * @returns {number} the same score from White's point of view
 */
export function toWhiteScore(score, searchStm) {
  return searchStm === WHITE ? score : -score;
}

/**
 * Formats a White-relative score: positive means White stands better,
 * negative means Black does, regardless of whose turn it is. Proven wins show
 * as +M3 / -M3, keeping the same sign convention.
 * @param {number} whiteScore from toWhiteScore()
 */
export function formatScore(whiteScore) {
  if (Math.abs(whiteScore) >= MATE_IN_MAX) {
    const plies = MATE - Math.abs(whiteScore);
    const moves = Math.max(1, Math.ceil(plies / 2));
    return `${whiteScore > 0 ? '+' : '-'}M${moves}`;
  }
  const v = whiteScore / 1024;
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
}
