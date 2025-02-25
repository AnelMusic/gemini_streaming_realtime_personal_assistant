/**
 * PCMProcessor - Handles PCM audio data streaming through AudioWorklet
 * 
 * This processor buffers incoming Float32Array audio data and outputs it to the audio hardware.
 */
class PCMProcessor extends AudioWorkletProcessor {
    /**
     * Initialize the processor
     */
    constructor() {
        super();
        this.buffer = new Float32Array();

        // Handle incoming audio data messages
        this.port.onmessage = (e) => {
            const newData = e.data;
            const newBuffer = new Float32Array(this.buffer.length + newData.length);
            newBuffer.set(this.buffer);
            newBuffer.set(newData, this.buffer.length);
            this.buffer = newBuffer;
        };
    }

    /**
     * Process audio data for each audio block
     * @param {Array} inputs - Input audio data (unused)
     * @param {Array} outputs - Output channels where audio will be written
     * @param {Object} parameters - Audio parameters (unused)
     * @returns {boolean} - Return true to keep the processor alive
     */
    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channelData = output[0];

        if (this.buffer.length >= channelData.length) {
            channelData.set(this.buffer.slice(0, channelData.length));
            this.buffer = this.buffer.slice(channelData.length);
            return true;
        }

        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
