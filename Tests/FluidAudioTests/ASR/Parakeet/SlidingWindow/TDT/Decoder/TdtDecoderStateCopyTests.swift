// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5. It tests
// that a copied decoder state shares no arrays with its original.

@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class TdtDecoderStateCopyTests: XCTestCase {

    private func decode(_ state: inout TdtDecoderState) async throws {
        let blank = ScriptedTdt.blank
        let joint = ScriptedJointModel { frame, last in
            frame == 0 && last == blank
                ? ScriptedJointStep(token: 8, probability: 0.7, durationBin: 1, topK: [(8, 5)])
                : ScriptedJointStep(token: blank, probability: 0.9, durationBin: 1, topK: [(blank, 6)])
        }
        _ = try await TdtDecoderV3(config: ScriptedTdt.config).decodeWithTimings(
            encoderOutput: try ScriptedTdt.encoderOutput(frames: 3), encoderSequenceLength: 3,
            actualAudioFrames: 3, decoderModel: ScriptedDecoderModel(), jointModel: joint,
            decoderState: &state)
    }

    private func values(_ state: TdtDecoderState) -> [Float] {
        [state.hiddenState, state.cellState].flatMap { array in (0..<array.count).map { array[$0].floatValue } }
    }

    /// The decoder writes its new LSTM state into the state's own arrays, so a decode on a copy
    /// must leave the original as it was, and a decode on the original must leave the copy.
    func testADecodeOnACopyLeavesTheOriginalAsItWas() async throws {
        var original = try TdtDecoderState()
        original.hiddenState[1] = 0.25
        original.cellState[1] = -0.5
        original.lastToken = 9
        let before = values(original)

        var copy = try TdtDecoderState(from: original)
        XCTAssertEqual(values(copy), before, "the copy starts where the original stands")
        XCTAssertEqual(copy.lastToken, 9)
        try await decode(&copy)
        XCTAssertNotEqual(values(copy), before, "the decode wrote the copy's arrays")
        XCTAssertEqual(values(original), before, "and left the original's alone")

        let copied = values(copy)
        try await decode(&original)
        XCTAssertNotEqual(values(original), before)
        XCTAssertEqual(values(copy), copied, "a decode on the original leaves the copy alone")
    }
}
