// Mirror of AzSnapshot in wasm/migoyugo_az_wasm.cpp.
//
// Every field is a single byte, so this is plain DataView-free indexing and
// there is no endianness question. The C++ side has a static_assert on the
// total size; SNAPSHOT_SIZE is checked against the module's reported size when
// the worker reports ready, so the two can never drift silently.

'use strict';

const AZ_SNAPSHOT_SIZE = 136;

const AZ_OFF = {
  version: 0,
  stm: 1,
  status: 2,
  winner: 3,
  moveCount: 4,
  lastMove: 5,
  canUndo: 6,
  // 7 reserved
  cells: 8,
  legal: 72,
};

const AZ_STATUS_PLAYING = 0;
const AZ_STATUS_OVER = 1;
const AZ_NO_SQUARE = 255;

// 0 empty, 1 W-Migo, 2 W-Yugo, 3 B-Migo, 4 B-Yugo - the same cell codes the
// NNUE engine's snapshot uses, so a board renderer can be shared.
function parseAzSnapshot(buffer) {
  const b = new Uint8Array(buffer);
  return {
    version: b[AZ_OFF.version],
    stm: b[AZ_OFF.stm],
    status: b[AZ_OFF.status],
    winner: b[AZ_OFF.winner],
    moveCount: b[AZ_OFF.moveCount],
    lastMove: b[AZ_OFF.lastMove] === AZ_NO_SQUARE ? -1 : b[AZ_OFF.lastMove],
    canUndo: b[AZ_OFF.canUndo] === 1,
    cells: b.slice(AZ_OFF.cells, AZ_OFF.cells + 64),
    legal: b.slice(AZ_OFF.legal, AZ_OFF.legal + 64),
  };
}

if (typeof self !== 'undefined') {
  self.AZ_SNAPSHOT_SIZE = AZ_SNAPSHOT_SIZE;
  self.AZ_STATUS_PLAYING = AZ_STATUS_PLAYING;
  self.AZ_STATUS_OVER = AZ_STATUS_OVER;
  self.AZ_NO_SQUARE = AZ_NO_SQUARE;
  self.parseAzSnapshot = parseAzSnapshot;
}
