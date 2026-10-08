// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5. It pins
// the decoder with no vocabulary bias to the unpatched decoder's output on a scripted decode.

@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class TdtDecoderV3ScriptedTests: XCTestCase {

    /// With no bias the hypothesis (tokens, timestamps, durations, probabilities) and the frames
    /// the joint was asked for are exactly what the unpatched v0.17.5 decoder gives on this script
    /// (the values were recorded from it), and top-K is never read. The script runs the main loop,
    /// a duration 0 emission and the same-frame guard, the inner blank loop, a token whose advance
    /// passes the last frame, and the last-chunk flush that decodes it again.
    func testWithNoBiasTheDecodeEqualsTheUnpatchedDecoders() async throws {
        let blank = ScriptedTdt.blank
        let joint = ScriptedJointModel { frame, last in
            switch (frame, last) {
            case (0, blank):
                return ScriptedJointStep(token: 8, probability: 0.7, durationBin: 0, topK: [(8, 5), (9, 4)])
            case (0, 8): return ScriptedJointStep(token: 9, probability: 0.6, durationBin: 0, topK: [(9, 5), (8, 4)])
            case (1, 9): return ScriptedJointStep(token: blank, probability: 0.9, durationBin: 2, topK: [(blank, 6)])
            case (3, 9): return ScriptedJointStep(token: 10, probability: 0.5, durationBin: 1, topK: [(10, 5)])
            case (4, 10): return ScriptedJointStep(token: 11, probability: 0.55, durationBin: 3, topK: [(11, 5)])
            case (5, 11): return ScriptedJointStep(token: 12, probability: 0.4, durationBin: 1, topK: [(12, 5)])
            default: return ScriptedJointStep(token: blank, probability: 0.9, durationBin: 1, topK: [(blank, 6)])
            }
        }
        var state = try TdtDecoderState()
        let frames = 6
        let hypothesis = try await TdtDecoderV3(config: ScriptedTdt.config).decodeWithTimings(
            encoderOutput: try ScriptedTdt.encoderOutput(frames: frames), encoderSequenceLength: frames,
            actualAudioFrames: frames, decoderModel: ScriptedDecoderModel(), jointModel: joint,
            decoderState: &state, isLastChunk: true)

        XCTAssertEqual(hypothesis.ySequence, [8, 9, 10, 11, 12])
        XCTAssertEqual(hypothesis.timestamps, [0, 0, 3, 5, 5])
        XCTAssertEqual(hypothesis.tokenDurations, [0, 1, 1, 3, 1])
        XCTAssertEqual(hypothesis.tokenConfidences, [0.7, 0.6, 0.5, 0.55, 0.4])
        XCTAssertEqual(joint.frames, [0, 0, 1, 3, 4, 5, 5, 4, 5, 5, 4, 5, 5, 4])
        XCTAssertEqual(state.lastToken, 12)
        XCTAssertFalse(joint.readLog.keys.contains("top_k_ids"), "no bias and no language read no top-K")
        XCTAssertFalse(joint.readLog.keys.contains("top_k_logits"))
    }
}
