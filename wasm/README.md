# GPlayer WASM Module

This directory contains a WebAssembly (WASM) build of the G-Rave player for playing Migoyugo. The module allows running the sophisticated Monte Carlo Tree Search (MCTS) algorithm with Rapid Action Value Estimation (RAVE) directly in web browsers.

## Features

- **High-performance AI**: Full G-Rave MCTS implementation
- **Web Worker support**: Asynchronous execution without blocking the UI
- **Memory efficient**: Stateless operation with proper memory management
- **Error handling**: Comprehensive error codes and messages

## Building

### Prerequisites

- Emscripten SDK (emcc, emcmake)
- CMake 3.18+
- The main rlcpp project built (for linking libraries)

### Build Steps

```bash
# From the wasm/ directory
chmod +x build.sh
./build.sh
```

This will create:
- `gplayer_wasm.js` - JavaScript glue code
- `gplayer_wasm.wasm` - WebAssembly binary

## Usage

### Basic Example

```javascript
// Create a Web Worker
const worker = new Worker('worker.js');

// Wait for initialization
worker.onmessage = function(e) {
    if (e.data.type === 'ready') {
        // Worker is ready, send a move request
        worker.postMessage({
            type: 'select_move',
            data: {
                actionsHistory: [0, 15, 42], // Example move history
                numSimulations: 1000,        // Minimum simulations
                durationMs: 5000,            // Time limit (5 seconds)
                minRefCount: 5,              // G-RAVE min reference count
                bias: 0.0,                   // Exploration bias
                saveIllegalActions: true    // Save illegal AMAF actions
            }
        });
    } else if (e.data.type === 'move_selected') {
        console.log('AI chose action:', e.data.data.action);
    } else if (e.data.type === 'error') {
        console.error('Error:', e.data.data.error);
    }
};
```

### API Reference

#### select_move Parameters

- `actionsHistory`: Array of integers representing the sequence of moves played so far
- `numSimulations`: Minimum number of MCTS simulations to run
- `durationMs`: Time limit in milliseconds for the search
- `minRefCount`: Minimum reference count for G-RAVE algorithm
- `bias`: Exploration bias parameter (typically 0.0)
- `saveIllegalActions`: Whether to save illegal actions in AMAF table

#### Return Values

- **Positive integer**: The chosen action (0-63 for 8x8 Migoyugo board)
- **Negative codes**: Error conditions:
  - `-1`: Invalid action history
  - `-2`: Game already ended
  - `-3`: Illegal action in history
  - `-4`: Attempted to step terminal state
  - `-5`: Generic error

## Migoyugo Game Rules

Migoyugo is played on an 8x8 board where players place pieces and can capture opponent pieces by surrounding them. Actions are encoded as integers 0-63, where:
- Row = action / 8
- Col = action % 8

## Performance Considerations

- MCTS simulations can be computationally intensive
- Use Web Workers to avoid blocking the main thread
- Typical search times: 1-5 seconds for good moves
- Memory usage scales with simulation count

## Integration

1. Copy `gplayer_wasm.js`, `gplayer_wasm.wasm`, and `worker.js` to your web project
2. Initialize the worker and wait for the 'ready' message
3. Send move requests as needed
4. Handle responses in the worker's onmessage handler

This module enables running professional-level game AI directly in web browsers for real-time gameplay and analysis.
