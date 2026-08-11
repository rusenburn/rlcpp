# Migoyugo in the browser

A WebAssembly build of the NNUE alpha-beta engine, plus the static site that
plays it. The engine is the same `NNUELayerStacksPlayerV2` the desktop UI uses:
a 384-feature quantized network evaluating a bitboard search with a
transposition table, principal variation search, killers, history and late move
reductions.

Measured in a browser-class runtime it reaches roughly **1.4M nodes/second**,
about 80% of the native build, because the SSE2 intrinsics in the evaluation
map one-to-one onto WebAssembly SIMD.

## Design: the rules live in C++

`wasm/migoyugo_wasm.cpp` exposes a small C ABI, and **every rule decision is
made on the C++ side** by `MigoyugoBB` — the bitboard engine that
`run/bench_migoyugo_bb.cpp` differentially tests against the reference
implementation over hundreds of thousands of plies. JavaScript never decides
whether a move is legal, what it captures, or whether the game is over; it asks
and renders the answer.

That is why the page can show things a hand-written JS board could not get
right: which squares are forbidden by the no-long-lines rule, which moves
promote to a Yugo, which win outright, and the exact four Yugos of a winning
line.

Two constraints shape the C++ entry points:

- **Nothing may throw.** Emscripten's default build turns a `throw` into an
  abort that kills the module permanently, so failures are return codes.
  `MigoyugoState` is never touched and `from_short` (which calls `std::stoi`)
  is never called.
- **Nothing may trust its caller.** `MigoyugoBB::do_move` guards legality with
  an `assert`, which `NDEBUG` removes. `Game::apply` validates range *and*
  legality before every move, and it is the single funnel that live moves,
  undo-replay and `load_moves` all pass through.

## Building

Requires the Emscripten SDK and an exported network.

```bash
source /path/to/emsdk/emsdk_env.sh
cd wasm
./build.sh
python3 -m http.server 8000 --directory ../build/wasm/web
# open http://localhost:8000/
```

A plain static server is enough — no COOP/COEP headers, no SharedArrayBuffer,
no threads. `file://` will not work, because both WebAssembly instantiation and
the `fetch()` of the network need `http://`.

Output in `build/wasm/web/`:

| File | |
|---|---|
| `index.html`, `styles.css` | the page |
| `app.js` | controller: modes, turn logic, rendering |
| `board.js` | persistent-DOM board renderer and animations |
| `sound.js` | WebAudio synthesis (no audio assets) |
| `snapshot.js` | the binary layout shared with C++ |
| `worker.js` | owns the module and the authoritative position |
| `migoyugo.js`, `migoyugo.wasm` | the engine |
| `migoyugo_nnue_v2.bin` | the 273 KB network |

The network is copied from `checkpoints/nnue_layerstacks_v2_weights.bin` at
build time. It is deliberately untracked (see `.gitignore`), so a missing file
fails the configure loudly rather than producing a site that 404s on its own
brain.

## Architecture

The worker owns the WebAssembly module **and** the authoritative game state;
the page is a pure view that talks to it over `postMessage`. During a search
the worker is blocked, which costs nothing: while the engine thinks it is not
the human's turn, so there is no move to validate, and hover, clicks and
animations all run on the main thread regardless.

Every request carries an **epoch**, bumped on new-game / undo / stop / mode
change. The worker echoes it and the page drops replies whose epoch is stale.
That single mechanism makes every race benign, including stopping mid-search:
the in-flight reply is discarded and the next move is simply never scheduled.
Engine-vs-engine runs one move per round trip from the page rather than as a
loop inside the worker, so stopping never needs to interrupt C++.

Because the position is fully reconstructible from a move list, a crashed
worker can be terminated, respawned and replayed with `mgy_load_moves`.

### Two things that will bite if you edit this

- **`mgy_snapshot()` is not a getter.** It rewrites the snapshot from the
  current position and returns its address. Cache the address and read it
  forever and the board freezes at the initial position — which looks like a
  dead UI, not an error.
- **Never cache a heap view.** `ALLOW_MEMORY_GROWTH` replaces `HEAPU8` on every
  grow and detaches the old one. Pointers into linear memory stay valid; views
  do not. Read `mod.HEAPU8` at each use.

## The C ABI

```c
int  mgy_init(const uint8_t* model, int len, int tt_mb);  // parses weights, builds the engine
void mgy_new_game(void);
int  mgy_play(int sq);                 // validates range AND legality
int  mgy_undo(int plies);
int  mgy_load_moves(const uint8_t* moves, int n);
int  mgy_bot_move(void);               // searches and plays
int  mgy_bot_suggest(void);            // searches without playing
void mgy_set_time_ms(int ms);          // difficulty; does not disturb the table
void mgy_set_tt_mb(int mb);            // resizing does clear it
void mgy_clear_tt(void);
const uint8_t* mgy_snapshot(void);  int mgy_snapshot_size(void);
const uint8_t* mgy_info(void);      int mgy_info_size(void);
```

Errors: `-1` no model, `-2` square out of range, `-3` illegal move, `-4` game
over, `-5` bad weights, `-6` model misaligned, `-7` nothing to undo.

The 416-byte snapshot and 32-byte info layouts are documented in
`web/snapshot.js` and asserted in `migoyugo_wasm.cpp`
(`static_assert(sizeof(Snapshot) == 416)`). The module also reports its own
sizes in the `ready` message so a browser-cached `worker.js` built against a
different layout fails loudly instead of misparsing.

## Testing

`run/test_wasm_api.cpp` links the same translation unit natively and drives the
whole ABI in a debugger — legality rejection, piline consistency, undo/replay
round-trips, promotion reporting, terminal detection, and the engine itself.
Build it with the normal native build and run:

```bash
./build/Release/bin/test_wasm_api checkpoints/nnue_layerstacks_v2_weights.bin
```

## The rules

Played on an 8×8 board. White moves first.

- **Placing.** On your turn, place a **Migo** (a plain stone) on any empty square.
- **Yugos.** Complete an unbroken line of **exactly four** of your own pieces
  and the stone you just placed becomes a **Yugo**, marked with a dot. Every
  Migo in those lines is removed. Yugos are permanent — they never move and are
  never captured. Multiple intersecting lines of exactly four are allowed.
- **No long lines.** You may never create an unbroken line of **more than four**
  of your own pieces. The page marks those squares.
- **Igo.** Form an unbroken line of exactly four of your own Yugos and you win
  instantly.
- **Wego.** If the player to move has no legal move, or the board is full, the
  game ends immediately and whoever has more Yugos wins. Equal Yugos is a draw.

A game can never repeat a position: Yugos are permanent, so a promoting move
strictly increases the Yugo count and any other move strictly increases the
occupied count.
