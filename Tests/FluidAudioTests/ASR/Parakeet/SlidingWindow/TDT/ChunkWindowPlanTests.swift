// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias-windows of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5. It
// tests the long-audio window plan, the windows ready before the recording ends, and the release's
// use of windows decoded ahead.

import Foundation
import XCTest
import os

@testable import FluidAudio

/// The window plan of the long-audio path and decode ahead (Koegaki spec 2026-11-03-mac-decode-ahead):
/// a window that is ready while the recording is still growing is exactly the window the finished
/// recording's plan decodes at that index, and the release takes a window's tokens from the caller when
/// the caller holds them and decodes it otherwise.
final class ChunkWindowPlanTests: XCTestCase {

    private let rate = ASRConstants.sampleRate
    private let frame = ASRConstants.samplesPerEncoderFrame

    // MARK: - Synthetic energy profiles

    /// A deterministic generator, so every profile is the same on every run.
    private struct Generator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state
        }
        mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
    }

    /// Speech-like audio: noise at a syllable-rate envelope that never goes quiet.
    private func speech(seconds: Double, seed: UInt64, level: Float = 0.1) -> [Float] {
        var generator = Generator(state: seed)
        let count = Int(seconds * Double(rate))
        return (0..<count).map { n in
            let envelope = 0.55 + 0.45 * sinf(Float(n) * 2 * .pi * 4 / Float(rate))
            return level * envelope * (generator.unit() * 2 - 1)
        }
    }

    /// `audio` with digital silence over each of `pauses` (start and length in seconds).
    private func silencing(_ audio: [Float], _ pauses: [(Double, Double)]) -> [Float] {
        var out = audio
        for (start, length) in pauses {
            let from = max(0, Int(start * Double(rate)))
            let to = min(out.count, Int((start + length) * Double(rate)))
            if from < to { for n in from..<to { out[n] = 0 } }
        }
        return out
    }

    /// The window targets of the Ultra layout, in seconds: one stride apart.
    private var strideSeconds: Double {
        let layout = ChunkProcessor(audioSamples: [Float](repeating: 0, count: rate))
            .chunkLayoutForTesting(melChunkContext: false, modelVersion: .ultra)
        return Double(layout.strideSamples) / Double(rate)
    }

    private func profiles() -> [(name: String, audio: [Float])] {
        let stride = strideSeconds
        var randomPauses: [(Double, Double)] = []
        var generator = Generator(state: 7)
        var at = 0.5
        while at < 80 {
            let length = 0.2 + 0.6 * Double(generator.unit())
            randomPauses.append((at, length))
            at += length + 1 + 3 * Double(generator.unit())
        }
        return [
            ("speech everywhere, 61 s", speech(seconds: 61, seed: 1)),
            (
                "a pause near every target, 75 s",
                silencing(speech(seconds: 75, seed: 2), (1...5).map { (Double($0) * stride - 1.3, 0.45) })
            ),
            (
                "pauses at the edges of the search radius, 66 s",
                silencing(
                    speech(seconds: 66, seed: 3),
                    (1...4).flatMap { [(Double($0) * stride - 4.05, 0.2), (Double($0) * stride + 3.9, 0.3)] })
            ),
            ("random pauses, 82 s", silencing(speech(seconds: 82, seed: 4), randomPauses)),
            (
                "quiet speech with one deep pause, 47 s",
                silencing(speech(seconds: 47, seed: 5, level: 0.004), [(25.5, 1.2)])
            ),
            // Lengths just past a window's end, so a short last window follows.
            ("just past the second window, 28 s", speech(seconds: 14.96 + stride + 0.05, seed: 6)),
            (
                "just past the third window, 41 s",
                silencing(speech(seconds: 14.96 + 2 * stride + 0.01, seed: 8), [(stride + 0.3, 0.5)])
            ),
        ]
    }

    private func plan(_ audio: [Float]) throws -> [ChunkWindow] {
        try ChunkProcessor(audioSamples: audio).plannedWindows(melChunkContext: false, modelVersion: .ultra)
    }

    private func ready(_ audio: [Float], dualDecodeArbitration: Bool = false) throws -> [ChunkWindow] {
        try ChunkProcessor.readyWindows(
            in: audio, melChunkContext: false, modelVersion: .ultra, dualDecodeArbitration: dualDecodeArbitration)
    }

    // MARK: - Plan stability

    /// For every profile and every prefix length (a frame apart, at a different offset within the frame
    /// each time), each window the prefix marks ready is the finished recording's window at its index,
    /// is not the finished recording's last window, has its audio complete inside the prefix, and stays
    /// ready as the recording grows.
    func testEveryReadyWindowIsTheFinishedPlansWindow() throws {
        for (name, audio) in profiles() {
            let full = try plan(audio)
            XCTAssertGreaterThan(full.count, 1, name)
            XCTAssertTrue(full.last?.isLastChunk == true && full.dropLast().allSatisfy { !$0.isLastChunk }, name)
            var previous: [ChunkWindow] = []
            var step = 0
            var length = ASRConstants.maxModelSamples - 3 * frame
            while length <= audio.count {
                let prefix = Array(audio[..<length])
                let marked = try ready(prefix)
                if length <= ASRConstants.maxModelSamples {
                    XCTAssertTrue(marked.isEmpty, "\(name): \(length) samples decode as one window, so none is ready")
                }
                XCTAssertEqual(
                    Array(marked.prefix(previous.count)), previous, "\(name): a ready window changed at \(length)")
                for window in marked {
                    guard window.index < full.count else {
                        XCTFail("\(name): window \(window.index) ready at \(length), past the finished plan")
                        continue
                    }
                    XCTAssertEqual(window, full[window.index], "\(name): window \(window.index) at \(length) samples")
                    XCTAssertFalse(full[window.index].isLastChunk, "\(name): the finished plan's last window was ready")
                    XCTAssertLessThan(window.sampleOffset + window.sampleCount, length, "\(name): incomplete audio")
                }
                previous = marked
                step += 1
                length = ASRConstants.maxModelSamples - 3 * frame + step * frame + (step * 97) % frame
            }
            XCTAssertEqual(
                try ready(audio), Array(full.dropLast()), "\(name): every window but the last is ready at the end")
        }
    }

    /// The plan is a pure function of the samples it is given: planning the same audio twice gives the
    /// same windows, and the windows tile the recording in order.
    func testThePlanIsInOrderAndRepeatable() throws {
        for (name, audio) in profiles() {
            let first = try plan(audio)
            XCTAssertEqual(first, try plan(audio), name)
            XCTAssertEqual(first.map(\.index), Array(first.indices), name)
            XCTAssertEqual(first.first?.sampleOffset, 0, name)
            for (left, right) in zip(first, first.dropFirst()) {
                XCTAssertLessThan(left.chunkStart, right.chunkStart, name)
                XCTAssertGreaterThan(left.sampleOffset + left.sampleCount, right.chunkStart, "\(name): windows overlap")
            }
            XCTAssertLessThanOrEqual((first.last.map { $0.sampleOffset + $0.sampleCount }) ?? 0, audio.count, name)
        }
    }

    /// Audio of one window or less (240,000 samples) decodes as one padded window, so nothing is ready
    /// however far its plan would go; dual decode arbitration plans its own windows, so none is ready.
    func testNothingIsReadyForOneWindowOrUnderDualDecodeArbitration() throws {
        let audio = speech(seconds: 40, seed: 9)
        XCTAssertTrue(try ready(Array(audio[..<ASRConstants.maxModelSamples])).isEmpty)
        XCTAssertTrue(try ready(Array(audio[..<(ASRConstants.maxModelSamples - frame)])).isEmpty)
        XCTAssertFalse(try ready(audio).isEmpty)
        XCTAssertTrue(try ready(audio, dualDecodeArbitration: true).isEmpty)
    }

    // MARK: - The release takes windows decoded ahead

    /// One window the release asked about, with the samples it was asked with.
    private struct Ask: Sendable {
        let window: ChunkWindow
        let samples: [Float]
    }

    /// Tokens a caller holds for `window`: one token a frame into the window, an id named by the index.
    private static func held(_ window: ChunkWindow) -> ChunkWindowTokens {
        ChunkWindowTokens(
            tokenIds: [100 + window.index],
            timestamps: [window.chunkStart / ASRConstants.samplesPerEncoderFrame + 1],
            confidences: [0.5 + Float(window.index) / 100], durations: [1])
    }

    /// Every window answered: the release asks for each window of its plan, in order, with exactly that
    /// window's samples, decodes none (the manager has no models, so a decode would throw), and merges
    /// the answers as it merges decoded windows.
    func testTheReleaseTakesEveryAnsweredWindowAndDecodesNone() async throws {
        let audio = [Float](repeating: 0, count: 40 * rate)
        let manager = AsrManager(config: .default)
        let asked = OSAllocatedUnfairLock(initialState: [Ask]())
        let processor = ChunkProcessor(audioSamples: audio)
        let result = try await processor.process(
            using: manager, startTime: Date(),
            decodedWindows: { window, samples in
                asked.withLock { $0.append(Ask(window: window, samples: samples)) }
                return Self.held(window)
            })
        let melChunkContext = await manager.melChunkContext
        let modelVersion = await manager.modelVersion
        let planned = try processor.plannedWindows(melChunkContext: melChunkContext, modelVersion: modelVersion)
        XCTAssertGreaterThan(planned.count, 2)
        let asks = asked.withLock { $0 }
        XCTAssertEqual(asks.map(\.window), planned, "asked for each window of the plan, in order")
        for ask in asks {
            let (window, samples) = (ask.window, ask.samples)
            XCTAssertEqual(samples, Array(audio[window.sampleOffset..<(window.sampleOffset + window.sampleCount)]))
        }
        XCTAssertEqual(result.tokenTimings?.map(\.tokenId), planned.map { 100 + $0.index }, "the answers, merged")
        XCTAssertEqual(result.tokenTimings?.map(\.confidence), planned.map { 0.5 + Float($0.index) / 100 })
    }

    /// A window the caller does not hold is decoded on a worker, as with no caller at all: with no
    /// models loaded that decode throws.
    func testAWindowNotAnsweredIsDecoded() async throws {
        let audio = [Float](repeating: 0, count: 40 * rate)
        let manager = AsrManager(config: .default)
        do {
            _ = try await ChunkProcessor(audioSamples: audio).process(
                using: manager, startTime: Date(),
                decodedWindows: { window, _ in window.index == 1 ? nil : Self.held(window) })
            XCTFail("window 1 was not answered, so it had to be decoded")
        } catch ASRError.notInitialized {
            // The decode of the one window not answered reached the manager, which has no models.
        } catch {
            XCTFail("the unanswered window's decode failed for another reason: \(error)")
        }
    }

    /// The manager's own entry points: no window is ready for one window of audio, and a window decode
    /// refuses a window outside the samples it is given.
    func testTheManagersEntryPoints() async throws {
        let manager = AsrManager(config: .default)
        let one = [Float](repeating: 0, count: ASRConstants.maxModelSamples)
        let none = try await manager.readyChunkWindows(in: one)
        XCTAssertTrue(none.isEmpty)
        // Past the samples, an empty window, an offset past the end, and an offset whose sum with the
        // count would overflow: each is refused as that, before any decode, with no trap.
        let past = ChunkWindow(
            index: 1, sampleOffset: 200_000, sampleCount: 239_360, contextSamples: 0,
            chunkStart: 200_000, isLastChunk: false, emitTokensAfterFrame: nil)
        let empty = ChunkWindow(
            index: 0, sampleOffset: 0, sampleCount: 0, contextSamples: 0,
            chunkStart: 0, isLastChunk: false, emitTokensAfterFrame: nil)
        let beyond = ChunkWindow(
            index: 0, sampleOffset: one.count + 1, sampleCount: 1, contextSamples: 0,
            chunkStart: 0, isLastChunk: false, emitTokensAfterFrame: nil)
        let overflowing = ChunkWindow(
            index: 0, sampleOffset: Int.max, sampleCount: 1, contextSamples: 0,
            chunkStart: 0, isLastChunk: false, emitTokensAfterFrame: nil)
        for (name, window) in [
            ("past the samples", past), ("empty", empty), ("beyond the end", beyond), ("overflowing", overflowing),
        ] {
            do {
                _ = try await manager.decodeChunkWindow(window, of: one)
                XCTFail("a window \(name) was decoded")
            } catch ASRError.processingFailed(let why) {
                XCTAssertTrue(why.contains("reads past the \(one.count) samples"), "\(name): \(why)")
            } catch {
                XCTFail("a window \(name) was refused for another reason: \(error)")
            }
        }
    }
}
