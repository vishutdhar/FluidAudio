// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias-windows of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5: the
// windows of a growing recording that are ready to decode, and one window's decode.

import Foundation

extension AsrManager {

    /// The windows of `samples`' long-audio plan that are not its last, for a recording that is still
    /// growing: each is exactly the window the finished recording's plan decodes at its index. None while
    /// `samples` are one window or less (240,000 samples, decoded as one padded window), and none under
    /// dual decode arbitration.
    public func readyChunkWindows(in samples: [Float]) throws -> [ChunkWindow] {
        try ChunkProcessor.readyWindows(
            in: samples, melChunkContext: melChunkContext, modelVersion: modelVersion,
            dualDecodeArbitration: dualDecodeArbitration)
    }

    /// One window of `samples` decoded on this manager, with its bias, from a fresh decoder state: the
    /// decode a long transcription gives the window on any of its workers. `samples` are the recording
    /// (at least through the window's last sample).
    public func decodeChunkWindow(
        _ window: ChunkWindow, of samples: [Float], language: Language? = nil
    ) async throws -> ChunkWindowTokens {
        // Each bound on its own, so a window built by hand with an absurd offset or count is refused
        // and never overflows the sum.
        guard window.sampleOffset >= 0, window.sampleOffset <= samples.count, window.sampleCount > 0,
            window.sampleCount <= samples.count - window.sampleOffset
        else {
            throw ASRError.processingFailed(
                "window \(window.index) reads past the \(samples.count) samples it was given")
        }
        let windowSamples = Array(samples[window.sampleOffset..<(window.sampleOffset + window.sampleCount)])
        let tokens = try await ChunkProcessor.decodeWindow(
            window, samples: windowSamples, using: self, decoderLayers: decoderLayerCount,
            maxModelSamples: ASRConstants.maxModelSamples, language: language)
        return ChunkWindowTokens(tokens)
    }
}
