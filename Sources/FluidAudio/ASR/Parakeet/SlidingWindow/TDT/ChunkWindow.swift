// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias-windows of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5: the
// long-audio window plan's windows, and decoding them while the recording is still growing.

import Foundation

/// One window of the long-audio path (audio over `ASRConstants.maxModelSamples`): exactly what
/// `ChunkProcessor` hands a worker to decode on a fresh decoder state. Two windows that are equal decode
/// the same samples the same way, so a window decoded while the recording was still growing can stand
/// in for the release's own decode of it when its samples are the same.
public struct ChunkWindow: Sendable, Hashable {
    /// The window's place in the plan, from 0.
    public let index: Int
    /// The first sample of the recording the window reads, and how many it reads (any prepended
    /// context or backfilled prefix included).
    public let sampleOffset: Int
    public let sampleCount: Int
    /// Prepended context samples the decoder skips.
    public let contextSamples: Int
    /// The sample the window's frames are counted from in the recording.
    public let chunkStart: Int
    /// Whether this is the plan's last window, decoded to the end of its audio.
    public let isLastChunk: Bool
    /// The recording-global encoder frame (`chunkStart` counted in frames of the whole recording, as the
    /// decode's `emitTokensAfterGlobalFrame`) before which emitted tokens are suppressed, when the window
    /// carries a backfilled prefix, or nil. The planner sets it; a window handed back unchanged keeps it.
    public let emitTokensAfterFrame: Int?

    public init(
        index: Int, sampleOffset: Int, sampleCount: Int, contextSamples: Int, chunkStart: Int,
        isLastChunk: Bool, emitTokensAfterFrame: Int?
    ) {
        self.index = index
        self.sampleOffset = sampleOffset
        self.sampleCount = sampleCount
        self.contextSamples = contextSamples
        self.chunkStart = chunkStart
        self.isLastChunk = isLastChunk
        self.emitTokensAfterFrame = emitTokensAfterFrame
    }
}

/// A window's decoded tokens, as the merge reads them: ids, global frame timestamps, confidences and
/// durations, one of each per token.
public struct ChunkWindowTokens: Sendable, Equatable {
    public let tokenIds: [Int]
    public let timestamps: [Int]
    public let confidences: [Float]
    public let durations: [Int]

    public init(tokenIds: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        self.tokenIds = tokenIds
        self.timestamps = timestamps
        self.confidences = confidences
        self.durations = durations
    }

    init(_ tokens: [ChunkProcessor.TokenWindow]) {
        tokenIds = tokens.map(\.token)
        timestamps = tokens.map(\.timestamp)
        confidences = tokens.map(\.confidence)
        durations = tokens.map(\.duration)
    }

    /// The tokens as the merge holds them; nil when the four lists do not line up.
    var tokenWindows: [ChunkProcessor.TokenWindow]? {
        guard tokenIds.count == timestamps.count, tokenIds.count == confidences.count,
            tokenIds.count == durations.count
        else { return nil }
        return tokenIds.indices.map {
            (token: tokenIds[$0], timestamp: timestamps[$0], confidence: confidences[$0], duration: durations[$0])
        }
    }
}

/// Asked by a long-audio transcription for each window before it decodes it, with the window and its
/// samples: the tokens the caller already decoded for exactly that window and those samples, on this
/// manager with its bias and the same language, or nil to decode it as usual.
public typealias ChunkWindowLookup = @Sendable (ChunkWindow, [Float]) -> ChunkWindowTokens?
