// Board renderer.
//
// Every square owns one persistent DOM node for its whole life, and each
// update diffs the new snapshot against the last one. The old demo rebuilt the
// grid with innerHTML on every change, which makes animation impossible: a
// node that is destroyed and recreated cannot transition, so a captured piece
// can only vanish.
//
// The board never reinterprets a square index. Flipping the view is a CSS
// transform on the grid (and a counter-rotation on the pieces so they stay
// upright); square 0 is always the top-left of the underlying position, which
// is what MigoyugoBB means by bit 0.

import {
  EMPTY, W_MIGO, W_YUGO, B_MIGO, B_YUGO,
  WHITE, isYugo, isWhite, squareName, rowOf, colOf,
} from './snapshot.js';

const CLEAR_STAGGER_MS = 45;

export class Board {
  #root;
  #squares = [];
  #pieces = new Array(64).fill(EMPTY);
  #onPick;
  #flipped = false;
  #interactive = false;
  #legal = null;

  constructor(root, onPick) {
    this.#root = root;
    this.#onPick = onPick;
    this.#build();
  }

  #build() {
    this.#root.innerHTML = '';

    const grid = document.createElement('div');
    grid.className = 'grid';

    for (let sq = 0; sq < 64; sq++) {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'sq';
      cell.dataset.sq = String(sq);
      cell.setAttribute('aria-label', squareName(sq));
      // Chequer using the underlying coordinates, so flipping cannot desync it.
      if (((rowOf(sq) + colOf(sq)) & 1) === 1) cell.classList.add('dark');

      const piece = document.createElement('span');
      piece.className = 'piece';
      piece.innerHTML = '<span class="dot"></span>';
      cell.appendChild(piece);

      const marker = document.createElement('span');
      marker.className = 'marker';
      cell.appendChild(marker);

      cell.addEventListener('click', () => {
        if (!this.#interactive) return;
        if (this.#legal && !this.#legal[sq]) {
          this.reject(sq);
          return;
        }
        this.#onPick(sq);
      });

      grid.appendChild(cell);
      this.#squares.push(cell);
    }

    const files = document.createElement('div');
    files.className = 'files';
    const ranks = document.createElement('div');
    ranks.className = 'ranks';
    for (let i = 0; i < 8; i++) {
      const f = document.createElement('span');
      f.textContent = String.fromCharCode(97 + i);
      files.appendChild(f);
      const r = document.createElement('span');
      r.textContent = String(8 - i);
      ranks.appendChild(r);
    }

    this.#root.append(ranks, grid, files);
    this.#grid = grid;
  }

  #grid = null;

  setFlipped(on) {
    this.#flipped = on;
    this.#root.classList.toggle('flipped', on);
  }

  setInteractive(on) {
    this.#interactive = on;
    this.#root.classList.toggle('interactive', on);
  }

  /**
   * @param {object} s parsed snapshot
   * @param {object} opts { animate, showPiline, showHints, hint }
   */
  render(s, opts = {}) {
    const { animate = true, showPiline = true, showHints = true, hint = -1 } = opts;
    this.#legal = s.legal;

    // Captures first, so the cleared pieces fade out from where they were
    // rather than being overwritten by the diff below.
    if (animate && s.cleared.length) {
      s.cleared.forEach((sq, i) => this.#clear(sq, i * CLEAR_STAGGER_MS));
    } else {
      for (const sq of s.cleared) this.#setPiece(sq, EMPTY, false);
    }

    for (let sq = 0; sq < 64; sq++) {
      const want = s.cells[sq];
      if (this.#pieces[sq] === want) continue;
      // A square that was just cleared is mid-animation; leave it alone unless
      // the snapshot says something new stands there.
      if (want === EMPTY && s.cleared.includes(sq) && animate) {
        this.#pieces[sq] = EMPTY;
        continue;
      }
      const dropping = animate && want !== EMPTY && sq === s.lastMove;
      this.#setPiece(sq, want, dropping);
    }

    for (let sq = 0; sq < 64; sq++) {
      const cell = this.#squares[sq];
      const legal = s.legal[sq] === 1;
      cell.classList.toggle('legal', legal && this.#interactive);
      cell.classList.toggle('promoting', showHints && legal && s.promoting[sq] === 1);
      cell.classList.toggle('winning', showHints && legal && s.winning[sq] === 1);
      cell.classList.toggle('last', sq === s.lastMove);
      cell.classList.toggle('hint', sq === hint);
      cell.classList.toggle('forbidden', showPiline && s.piline[s.stm][sq] === 1);
      cell.classList.remove('win-line');
      cell.disabled = !this.#interactive || !legal;
    }

    for (const sq of s.winLine) this.#squares[sq].classList.add('win-line');

    if (animate && s.lastPromotion && s.lastMove >= 0) {
      this.#pulse(this.#squares[s.lastMove], 'promoted');
    }
  }

  reject(sq) {
    this.#pulse(this.#squares[sq], 'refused');
  }

  #pulse(cell, cls) {
    cell.classList.remove(cls);
    // Force a reflow so re-adding the class restarts the animation.
    void cell.offsetWidth;
    cell.classList.add(cls);
    cell.addEventListener('animationend', () => cell.classList.remove(cls), { once: true });
  }

  // A square's appearance is a pure function of its class list, and only these
  // two methods ever write it.
  //
  // There are deliberately NO animation listeners and no timers here. An
  // earlier version stripped the classes off a captured stone on `animationend`
  // and had to cancel that cleanup when a new stone landed on the same square.
  // That is unwinnable: mutating `animation-delay` or removing the class fires
  // `animationcancel` rather than `animationend`, so the pending handler
  // survived and later fired on the *new* piece's drop animation, erasing a
  // stone the engine had already placed. It showed up as pieces occasionally
  // not appearing when the engine moved fast - usually when it had found a
  // forced win and was replying instantly.
  //
  // Instead `leaving` carries its own appearance and ends at opacity 0 via
  // animation-fill-mode: forwards, so the final state needs no cleanup at all.
  // Whatever order these two run in, the class list afterwards is correct.
  #setPiece(sq, code, dropping) {
    const piece = this.#squares[sq].querySelector('.piece');
    this.#pieces[sq] = code;

    piece.style.animationDelay = '';
    piece.classList.remove('leaving');
    piece.classList.toggle('white', code !== EMPTY && isWhite(code));
    piece.classList.toggle('black', code !== EMPTY && !isWhite(code));
    piece.classList.toggle('yugo', code !== EMPTY && isYugo(code));
    piece.classList.toggle('present', code !== EMPTY);

    if (dropping) {
      piece.classList.remove('dropping');
      void piece.offsetWidth; // reflow, so re-adding restarts the animation
      piece.classList.add('dropping');
    }
  }

  #clear(sq, delayMs) {
    const piece = this.#squares[sq].querySelector('.piece');
    if (!piece.classList.contains('present')) return;

    this.#pieces[sq] = EMPTY;
    piece.classList.remove('present', 'dropping');
    piece.style.animationDelay = `${delayMs}ms`;
    piece.classList.add('leaving');
  }
}
