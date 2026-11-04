// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias-windows of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5. It
// checks, on a real model and real speech, that a release whose windows were decoded ahead is the
// one-shot decode.

import AVFoundation
import Foundation
import XCTest
import os

@testable import FluidAudio

/// Decode ahead on a real model (Koegaki spec 2026-11-03-mac-decode-ahead, section 6): the three speech
/// fixtures joined (about 58 s, four windows of the Ultra layout) are decoded one shot, then released
/// with their first one, two and three windows answered from `decodeChunkWindow` and the rest decoded
/// fresh, and the results compare token by token (ids, times, confidences, text). Two windows are also
/// decoded at once on the same manager, against their sequential decodes. Koegaki's own gate
/// (`DecodeAheadIdentityTests`, LibriSpeech holds of 16 to 300 s, bit for bit) is the broader proof;
/// this keeps the fork honest on its own. Not shown here: a seam-gap repair that actually runs (it needs
/// the decoder to miss half a second of speech between two tokens, which these fixtures do not provoke);
/// the repair runs after the merge, on the same manager, whether a window was answered or decoded.
/// Skipped without a model folder: set FLUIDAUDIO_MODELS_DIR to a folder holding the Ultra set
/// (Preprocessor, Encoder, Decoder, JointDecision and the vocabulary).
final class ChunkWindowDecodeAheadIdentityTests: XCTestCase {

    private static let fixtures = [
        "01-validation-request-21.4s", "02-release-readiness-19.8s", "03-diff-explanation-16.9s",
    ]

    private func loadedManager() async throws -> AsrManager {
        guard let folder = ProcessInfo.processInfo.environment["FLUIDAUDIO_MODELS_DIR"] else {
            throw XCTSkip("set FLUIDAUDIO_MODELS_DIR to a folder holding the Ultra set")
        }
        let models = try AsrModels.loadLocal(
            from: URL(fileURLWithPath: folder, isDirectory: true), version: .ultra, encoderPrecision: .int8,
            encoderComputeUnits: .cpuAndNeuralEngine)
        let manager = AsrManager(config: .default)
        try await manager.loadModels(models)
        return manager
    }

    private static func samples(of name: String) throws -> [Float] {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: "wav")
                ?? Bundle.module.url(forResource: name, withExtension: "wav")
        else { throw XCTSkip("fixture \(name) missing from test bundle") }
        let file = try AVAudioFile(forReading: url)
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { throw XCTSkip("could not allocate a buffer for \(name)") }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else { throw XCTSkip("\(name) is not float PCM") }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    /// The three fixtures one after another, a third of a second of silence between them.
    private static func recording() throws -> [Float] {
        let gap = [Float](repeating: 0, count: ASRConstants.sampleRate / 3)
        return try fixtures.map(samples(of:)).joined(separator: gap).map { $0 }
    }

    private struct Tokens: Equatable {
        let ids: [Int]
        let starts: [TimeInterval]
        let ends: [TimeInterval]
        let confidences: [Float]
        let text: String
        init(_ result: ASRResult) {
            let timings = result.tokenTimings ?? []
            ids = timings.map(\.tokenId)
            starts = timings.map(\.startTime)
            ends = timings.map(\.endTime)
            confidences = timings.map(\.confidence)
            text = result.text
        }
    }

    func testTheReleaseWithWindowsDecodedAheadIsTheOneShotDecode() async throws {
        let manager = try await loadedManager()
        let audio = try Self.recording()
        XCTAssertGreaterThan(audio.count, 3 * ASRConstants.maxModelSamples / 2, "the recording spans several windows")

        let oneShot = Tokens(try await ChunkProcessor(audioSamples: audio).process(using: manager, startTime: Date()))
        XCTAssertFalse(oneShot.ids.isEmpty, "the recording decodes to words")

        let ready = try await manager.readyChunkWindows(in: audio)
        XCTAssertGreaterThanOrEqual(ready.count, 3, "three windows are ready before the last: \(ready.count)")
        var ahead: [ChunkWindow: ChunkWindowTokens] = [:]
        for window in ready {
            ahead[window] = try await manager.decodeChunkWindow(window, of: audio)
        }

        // The first one, two and three windows answered, the rest decoded fresh at the release.
        let answers = ahead
        for answered in 1...min(3, ready.count) {
            let held = Set(ready.prefix(answered))
            let asked = OSAllocatedUnfairLock(initialState: Set<ChunkWindow>())
            let released = Tokens(
                try await ChunkProcessor(audioSamples: audio).process(
                    using: manager, startTime: Date(),
                    decodedWindows: { window, samples in
                        asked.withLock { _ = $0.insert(window) }
                        guard held.contains(window) else { return nil }
                        XCTAssertEqual(
                            samples, Array(audio[window.sampleOffset..<(window.sampleOffset + window.sampleCount)]))
                        return answers[window]
                    }))
            XCTAssertEqual(released, oneShot, "\(answered) window(s) answered: the release is not the one-shot decode")
            XCTAssertTrue(
                held.isSubset(of: asked.withLock { $0 }), "\(answered) window(s) answered: each was asked for")
        }

        // Two windows decoded at once on the same manager give their sequential decodes.
        if ready.count >= 2 {
            async let first = manager.decodeChunkWindow(ready[0], of: audio)
            async let second = manager.decodeChunkWindow(ready[1], of: audio)
            let (a, b) = try await (first, second)
            XCTAssertEqual(a, ahead[ready[0]], "window 0 decoded beside window 1")
            XCTAssertEqual(b, ahead[ready[1]], "window 1 decoded beside window 0")
        }
        print(
            "decode ahead identity: \(ready.count) windows ready of the recording's plan, \(oneShot.ids.count) tokens, text: \(oneShot.text.prefix(80))"
        )
    }
}
