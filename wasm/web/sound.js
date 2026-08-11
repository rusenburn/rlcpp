// Synthesised sound effects.
//
// Everything is generated with WebAudio oscillators rather than shipped as
// audio files: nothing to download, nothing to license, and the whole module
// is smaller than a single short .wav.
//
// Browsers refuse to start an AudioContext until the user has interacted with
// the page, so the context is created lazily on the first gesture and every
// play() is a no-op until then. That is why none of this needs a "click to
// enable audio" prompt.

export class Sound {
  #ctx = null;
  #master = null;
  #enabled = true;

  get enabled() { return this.#enabled; }

  setEnabled(on) {
    this.#enabled = on;
    if (on) this.resume();
    else if (this.#master) this.#master.gain.value = 0;
  }

  // Call from a real user gesture (a click), otherwise the context stays
  // suspended and nothing is audible.
  // Audio must never be able to break the game: a machine with no audio device
  // can throw from the AudioContext constructor, and this is called from the
  // click handler that also submits the move.
  resume() {
    if (!this.#enabled || this.#broken) return;
    try {
      if (!this.#ctx) {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) { this.#broken = true; return; }
        this.#ctx = new Ctx();
        this.#master = this.#ctx.createGain();
        this.#master.gain.value = 0.22;
        this.#master.connect(this.#ctx.destination);
      }
      if (this.#ctx.state === 'suspended') this.#ctx.resume();
      if (this.#master) this.#master.gain.value = 0.22;
    } catch (e) {
      console.warn('audio unavailable, continuing without sound:', e);
      this.#broken = true;
    }
  }

  #broken = false;

  // One enveloped oscillator. `type` picks the timbre; the short attack and
  // exponential decay are what keep these from sounding like test tones.
  #tone({ freq, dur = 0.12, type = 'sine', gain = 1, delay = 0, sweepTo = null }) {
    if (!this.#enabled || this.#broken || !this.#ctx || this.#ctx.state !== 'running') return;
    try {
    const t0 = this.#ctx.currentTime + delay;

    const osc = this.#ctx.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t0);
    if (sweepTo) osc.frequency.exponentialRampToValueAtTime(sweepTo, t0 + dur);

    const env = this.#ctx.createGain();
    env.gain.setValueAtTime(0.0001, t0);
    env.gain.exponentialRampToValueAtTime(gain, t0 + 0.008);
    env.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);

    osc.connect(env).connect(this.#master);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
    } catch (e) { this.#broken = true; }
  }

  // A short filtered noise burst, for the capture "sweep".
  #noise({ dur = 0.14, gain = 0.5, delay = 0, freq = 1800 }) {
    if (!this.#enabled || this.#broken || !this.#ctx || this.#ctx.state !== 'running') return;
    try {
    const t0 = this.#ctx.currentTime + delay;
    const frames = Math.floor(this.#ctx.sampleRate * dur);
    const buffer = this.#ctx.createBuffer(1, frames, this.#ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < frames; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / frames);

    const src = this.#ctx.createBufferSource();
    src.buffer = buffer;

    const filter = this.#ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(freq, t0);
    filter.frequency.exponentialRampToValueAtTime(freq * 0.35, t0 + dur);

    const env = this.#ctx.createGain();
    env.gain.setValueAtTime(gain, t0);
    env.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);

    src.connect(filter).connect(env).connect(this.#master);
    src.start(t0);
    } catch (e) { this.#broken = true; }
  }

  place(isWhite) {
    // A soft wooden knock; the two colours differ slightly so you can hear
    // whose stone landed without looking.
    this.#tone({ freq: isWhite ? 330 : 262, dur: 0.09, type: 'triangle', gain: 0.5, sweepTo: isWhite ? 220 : 175 });
    this.#tone({ freq: isWhite ? 990 : 786, dur: 0.04, type: 'sine', gain: 0.12 });
  }

  promote() {
    // A rising third: something was gained and it is permanent.
    this.#tone({ freq: 523.25, dur: 0.13, type: 'triangle', gain: 0.5 });
    this.#tone({ freq: 659.25, dur: 0.16, type: 'triangle', gain: 0.45, delay: 0.07 });
    this.#tone({ freq: 987.77, dur: 0.22, type: 'sine', gain: 0.3, delay: 0.14 });
  }

  capture(count) {
    // One brushed tick per stone removed, capped so a 12-stone clear does not
    // turn into a machine gun.
    const n = Math.min(count, 6);
    for (let i = 0; i < n; i++) {
      this.#noise({ dur: 0.1, gain: 0.35, delay: i * 0.045, freq: 2200 - i * 150 });
    }
  }

  illegal() {
    this.#tone({ freq: 180, dur: 0.11, type: 'square', gain: 0.22, sweepTo: 120 });
  }

  win() {
    [523.25, 659.25, 783.99, 1046.5].forEach((f, i) =>
      this.#tone({ freq: f, dur: 0.3, type: 'triangle', gain: 0.42, delay: i * 0.1 }));
  }

  lose() {
    [392, 329.63, 261.63].forEach((f, i) =>
      this.#tone({ freq: f, dur: 0.34, type: 'sine', gain: 0.4, delay: i * 0.13 }));
  }

  draw() {
    [440, 415.3].forEach((f, i) =>
      this.#tone({ freq: f, dur: 0.3, type: 'sine', gain: 0.35, delay: i * 0.14 }));
  }
}
