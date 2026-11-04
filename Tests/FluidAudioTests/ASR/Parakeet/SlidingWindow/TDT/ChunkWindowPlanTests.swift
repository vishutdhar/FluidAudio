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

    // MARK: - The recordings

    /// The fixtures joined in every order of two and of three (36.7 to 58.1 s): real speech with its own
    /// pauses, which fall near a different target in each order.
    private func recordings() throws -> [(name: String, audio: [Float])] {
        try ChunkWindowFixtures.joinedRecordings()
    }

    private func plan(_ audio: [Float]) throws -> [ChunkWindow] {
        try ChunkProcessor(audioSamples: audio).plannedWindows(melChunkContext: false, modelVersion: .ultra)
    }

    private func ready(_ audio: [Float], dualDecodeArbitration: Bool = false) throws -> [ChunkWindow] {
        try ChunkProcessor.readyWindows(
            in: audio, melChunkContext: false, modelVersion: .ultra, dualDecodeArbitration: dualDecodeArbitration)
    }

    // MARK: - Plan stability

    /// For every recording and every prefix length (a frame apart, at a different offset within the frame
    /// each time), each window the prefix marks ready is the finished recording's window at its index,
    /// is not the finished recording's last window, has its audio complete inside the prefix, and stays
    /// ready as the recording grows.
    func testEveryReadyWindowIsTheFinishedPlansWindow() throws {
        for (name, audio) in try recordings() {
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
        for (name, audio) in try recordings() {
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
        let audio = try ChunkWindowFixtures.allThree()
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

    /// Tokens a caller holds for `window`: one every 8 frames from a frame into the window through its own
    /// stride (never into the next window's region or past its own audio), every id named by the index. The
    /// tokens are dense on purpose: the release's seam-gap repair probes a gap of 1.5 s or more between two
    /// tokens that holds speech, and these tests run on real speech with a manager that has no models, so
    /// no gap may reach it.
    private static func held(_ window: ChunkWindow) -> ChunkWindowTokens {
        let frame = ASRConstants.samplesPerEncoderFrame
        let first = window.chunkStart / frame + 1
        let last = min(first + 144, (window.sampleOffset + window.sampleCount) / frame - 2)
        let frames = Array(stride(from: first, through: last, by: 8))
        return ChunkWindowTokens(
            tokenIds: frames.map { _ in 100 + window.index },
            timestamps: frames,
            confidences: frames.map { _ in 0.5 + Float(window.index) / 100 },
            durations: frames.map { _ in 1 })
    }

    /// Every window answered: the release asks for each window of its plan, in order, with exactly that
    /// window's samples, decodes none (the manager has no models, so a decode would throw), and merges
    /// the answers as it merges decoded windows.
    func testTheReleaseTakesEveryAnsweredWindowAndDecodesNone() async throws {
        let audio = try ChunkWindowFixtures.allThree()
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
        XCTAssertEqual(
            result.tokenTimings?.map(\.tokenId), planned.flatMap { Self.held($0).tokenIds }, "the answers, merged")
        XCTAssertEqual(result.tokenTimings?.map(\.confidence), planned.flatMap { Self.held($0).confidences })
    }

    /// A window the caller does not hold is decoded on a worker, as with no caller at all: with no
    /// models loaded that decode throws.
    func testAWindowNotAnsweredIsDecoded() async throws {
        let audio = try ChunkWindowFixtures.allThree()
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
        let one = Array(
            try ChunkWindowFixtures.samples(of: ChunkWindowFixtures.names[0])[..<ASRConstants.maxModelSamples])
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
