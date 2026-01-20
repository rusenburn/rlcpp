// Web Worker for GPlayer WASM module
// This worker handles the computationally intensive MCTS search
// without blocking the main browser thread

// let Module = null;

// Initialize the WASM module
let isModuleReady = false;

// Configure Module before loading WASM script
if (typeof Module === 'undefined') {
    Module = {};
}

Module.onRuntimeInitialized = function () {
    // WASM module is ready
    isModuleReady = true;
    postMessage({ type: 'ready' });
};

// Handle stdout from WASM (for debugging)
Module.print = function (text) {
    console.log('WASM stdout:', text);
};

// Handle stderr from WASM (for debugging)
Module.printErr = function (text) {
    console.error('WASM stderr:', text);
};

// Load the WASM module after configuring Module
importScripts('gplayer_wasm.js');

// Listen for messages from the main thread
self.onmessage = function (e) {
    const { type, data } = e.data;

    switch (type) {
        case 'select_move':
            handleSelectMove(data);
            break;
        default:
            console.error('Unknown message type:', type);
    }
};

function handleSelectMove(data) {
    try {
        // Check if WASM module is ready
        if (!isModuleReady || typeof Module.ccall !== 'function') {
            postMessage({
                type: 'error',
                data: { error: 'WASM module not ready yet', code: -100 }
            });
            return;
        }

        const {
            actionsHistory,
            numSimulations,
            durationMs,
            minRefCount,
            bias,
            saveIllegalActions
        } = data;


        const history32 = new Int32Array(actionsHistory);
        const ptr = Module._malloc(history32.length * 4);

        HEAP32.set(history32, ptr >> 2);

        const result = Module.ccall(
            'select_move',
            'number',
            ['number', 'number', 'number', 'number', 'number', 'number', 'number'],
            [ptr, actionsHistory.length, numSimulations, durationMs, minRefCount, bias, saveIllegalActions ? 1 : 0]
        );

        Module._free(ptr);
        console.log('Worker: select_move returned:', result, typeof result);

        // Send result back to main thread
        if (result >= 0) {
            postMessage({
                type: 'move_selected',
                data: { action: result }
            });
        } else {
            // Handle error codes
            const errorMessages = {
                [-1]: 'Invalid action history',
                [-2]: 'Game already ended',
                [-3]: 'Illegal action in history',
                [-4]: 'Attempted to step terminal state',
                [-5]: 'Generic error',
                [-6]: 'Invalid history length',
                [-7]: 'Invalid action value'
            };

            postMessage({
                type: 'error',
                data: { error: errorMessages[result] || `Unknown error: ${result}`, code: result }
            });
        }

    } catch (error) {
        postMessage({
            type: 'error',
            data: { error: error.message, code: -99 }
        });
    }
}
