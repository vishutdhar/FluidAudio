// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5. It tests
// the decode-time vocabulary bias through `TdtDecoderV3.decodeWithTimings` with scripted models.

@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

private let blank = ScriptedTdt.blank

/// A blank step that advances `durationBin` frames.
private func blankStep(_ durationBin: Int = 1) -> ScriptedJointStep {
    ScriptedJointStep(token: blank, probability: 0.9, durationBin: durationBin, topK: [(blank, 6), (8, 1)])
}

final class TdtVocabularyBiasDecoderTests: XCTestCase {

    // The toy piece table of TdtVocabularyBiasTests (a leading space marks a word start).
    private let vocab: [Int: String] = [
        8: " the", 9: " Ch", 10: " Si", 11: "ob", 12: "han", 13: "av", 16: " Ter", 17: "min", 18: " termin",
        19: "al", 8192: "<blank>",
    ]
    private let siobhan = CustomVocabularyTerm(text: "Siobhan", tokenIds: [10, 11, 12])
    private let terminal = CustomVocabularyTerm(text: "Terminal", tokenIds: [16, 17, 19])

    private func bias(_ terms: [CustomVocabularyTerm], boost: Float = 1, minLetters: Int = 2) -> TdtVocabularyBias {
        TdtVocabularyBias(
            terms: terms, vocabulary: vocab, blankId: blank, boost: boost, shape: .contextGraph,
            freshStartMinLetters: minLetters, holdFrameOnShorterFlip: true)!
    }

    private func decode(
        frames: Int, bias: TdtVocabularyBias?, emitAfter: Int? = nil,
        script: @escaping (_ frame: Int, _ lastToken: Int) -> ScriptedJointStep
    ) async throws -> (hypothesis: TdtHypothesis, joint: ScriptedJointModel) {
        let joint = ScriptedJointModel(script: script)
        var state = try TdtDecoderState()
        let hypothesis = try await TdtDecoderV3(config: ScriptedTdt.config).decodeWithTimings(
            encoderOutput: try ScriptedTdt.encoderOutput(frames: frames), encoderSequenceLength: frames,
            actualAudioFrames: frames, decoderModel: ScriptedDecoderModel(), jointModel: joint,
            decoderState: &state, emitTokensAfterGlobalFrame: emitAfter, vocabularyBias: bias)
        return (hypothesis, joint)
    }

    /// At the main site the plain " Ch" gives way to " Si", which opens Siobhan: the engaged token
    /// is emitted at the frame the plain one would have been, with its top-K softmax.
    func testFlipAtTheMainSiteEmitsTheEngagedTokenWithItsTopKSoftmax() async throws {
        let logits: [Float] = [5, 3.5, 1]
        let (hypothesis, joint) = try await decode(frames: 3, bias: bias([siobhan])) { frame, _ in
            frame == 0
                ? ScriptedJointStep(
                    token: 9, probability: 0.9, durationBin: 1,
                    topK: [(9, logits[0]), (10, logits[1]), (blank, logits[2])])
                : blankStep()
        }
        XCTAssertEqual(hypothesis.ySequence, [10])
        XCTAssertEqual(hypothesis.timestamps, [0])
        XCTAssertEqual(hypothesis.tokenDurations, [1])
        XCTAssertEqual(hypothesis.tokenConfidences.count, 1)
        XCTAssertEqual(
            hypothesis.tokenConfidences.first ?? -1, ScriptedTdt.topKSoftmax(3.5, in: logits), accuracy: 1e-6)
        XCTAssertTrue(joint.readLog.keys.contains("top_k_ids"), "a bias reads the joint's top-K")
    }

    /// The same step with a blank argmax: a blank is never overridden, however strong the bias, so
    /// silence never fills with words.
    func testABlankArgmaxEmitsNothingExtra() async throws {
        let (hypothesis, joint) = try await decode(frames: 3, bias: bias([siobhan], boost: 6, minLetters: 0)) {
            frame, _ in
            frame == 0
                ? ScriptedJointStep(
                    token: blank, probability: 0.9, durationBin: 1,
                    topK: [(blank, 5), (10, 4.9), (9, 4)])
                : blankStep()
        }
        XCTAssertEqual(hypothesis.ySequence, [])
        XCTAssertTrue(joint.readLog.keys.contains("top_k_ids"), "the bias was active")
    }

    /// In the inner blank loop the plain " termin" gives way to " Ter", which opens Terminal; it is
    /// shorter, so the decoder holds the frame and reads it again.
    func testFlipAtTheInnerBlankLoopSiteHoldsTheFrame() async throws {
        let (hypothesis, joint) = try await decode(frames: 4, bias: bias([terminal])) { frame, last in
            switch (frame, last) {
            case (0, _): return blankStep(1)
            case (1, blank):
                return ScriptedJointStep(token: 18, probability: 0.8, durationBin: 2, topK: [(18, 5), (16, 3.5)])
            case (1, 16):
                return ScriptedJointStep(token: 17, probability: 0.8, durationBin: 1, topK: [(17, 5), (16, 1)])
            default: return blankStep()
            }
        }
        XCTAssertEqual(hypothesis.ySequence.first, 16, "the inner site flipped \" termin\" to \" Ter\"")
        XCTAssertEqual(hypothesis.timestamps.first, 1)
        XCTAssertEqual(Array(joint.frames.prefix(3)), [0, 1, 1], "the frame was held after the shorter flip")
        XCTAssertEqual(hypothesis.ySequence, [16, 17])
    }

    /// Hold-frame at the main site: after a flip to a shorter piece the next joint call reads the
    /// same encoder frame; after a flip to a piece as long, the frame advances by the model's duration.
    func testHoldFrameOnlyAfterAShorterFlip() async throws {
        let (held, heldJoint) = try await decode(frames: 4, bias: bias([terminal])) { frame, last in
            switch (frame, last) {
            case (0, blank):
                return ScriptedJointStep(token: 18, probability: 0.8, durationBin: 2, topK: [(18, 5), (16, 3.5)])
            case (0, 16):
                return ScriptedJointStep(token: 17, probability: 0.8, durationBin: 1, topK: [(17, 5), (8, 1)])
            case (1, 17):
                return ScriptedJointStep(token: 19, probability: 0.8, durationBin: 1, topK: [(19, 5), (8, 1)])
            default: return blankStep()
            }
        }
        XCTAssertEqual(Array(heldJoint.frames.prefix(2)), [0, 0], "\" termin\" (6 letters) to \" Ter\" (3) holds")
        XCTAssertEqual(held.ySequence, [16, 17, 19])
        XCTAssertEqual(held.timestamps, [0, 0, 1])

        let (moved, movedJoint) = try await decode(frames: 4, bias: bias([siobhan])) { frame, last in
            switch (frame, last) {
            case (0, blank):
                return ScriptedJointStep(token: 9, probability: 0.8, durationBin: 2, topK: [(9, 5), (10, 3.5)])
            default: return blankStep()
            }
        }
        XCTAssertEqual(Array(movedJoint.frames.prefix(2)), [0, 2], "\" Ch\" to \" Si\" (2 letters each) advances")
        XCTAssertEqual(moved.ySequence, [10])
        XCTAssertEqual(moved.tokenDurations, [2])
    }

    /// Hold-frame wins over the decoder's same-frame guard: when a token was already emitted at the
    /// frame with duration 0, a later flip to a shorter piece at that frame still holds it, so the
    /// audio the shorter piece did not spell is decoded.
    func testHoldFrameAfterAnEmissionAtTheSameFrame() async throws {
        let (hypothesis, joint) = try await decode(frames: 4, bias: bias([terminal])) { frame, last in
            switch (frame, last) {
            case (0, blank):
                return ScriptedJointStep(token: 8, probability: 0.8, durationBin: 0, topK: [(8, 5), (9, 1)])
            case (0, 8):
                return ScriptedJointStep(token: 18, probability: 0.8, durationBin: 2, topK: [(18, 5), (16, 3.5)])
            case (0, 16):
                return ScriptedJointStep(token: 17, probability: 0.8, durationBin: 1, topK: [(17, 5), (8, 1)])
            default: return blankStep()
            }
        }
        XCTAssertEqual(Array(hypothesis.ySequence.prefix(2)), [8, 16], "\" termin\" flipped to \" Ter\"")
        XCTAssertEqual(Array(joint.frames.prefix(3)), [0, 0, 0], "the frame is held after the shorter flip")
        XCTAssertEqual(hypothesis.ySequence, [8, 16, 17])
        XCTAssertEqual(hypothesis.timestamps, [0, 0, 0])
    }

    /// A token before the emission cutoff is suppressed, yet it reached the LSTM, so it advances the
    /// graph: after a suppressed " Si" the next arc of Siobhan is still boosted.
    func testASuppressedTokenAdvancesTheGraph() async throws {
        let (hypothesis, _) = try await decode(frames: 4, bias: bias([siobhan]), emitAfter: 1) { frame, last in
            switch (frame, last) {
            case (0, blank):
                return ScriptedJointStep(token: 10, probability: 0.8, durationBin: 1, topK: [(10, 5), (9, 4)])
            case (1, 10):
                return ScriptedJointStep(token: 13, probability: 0.8, durationBin: 1, topK: [(13, 5), (11, 2.5)])
            default: return blankStep()
            }
        }
        XCTAssertEqual(hypothesis.suppressedTokens, [10])
        XCTAssertEqual(hypothesis.ySequence, [11], "\"ob\" continues the suppressed \" Si\"")
    }

    /// `makeWorkerClone` carries the bias, so the chunk workers of long audio decode with it too.
    func testWorkerClonesCarryTheBias() async throws {
        let models = AsrModels(
            encoder: nil, preprocessor: ScriptedDecoderModel(), decoder: ScriptedDecoderModel(),
            joint: ScriptedDecoderModel(), configuration: MLModelConfiguration(), vocabulary: vocab, version: .ultra)
        let manager = AsrManager(config: .default)
        try await manager.loadModels(models)
        let built = await manager.makeVocabularyBias(
            terms: [siobhan], boost: 0.5, shape: .contextGraph, freshStartMinLetters: 2, holdFrameOnShorterFlip: true)
        XCTAssertNotNil(built)
        await manager.setVocabularyBias(built)
        let clone = await manager.makeWorkerClone()
        XCTAssertNotNil(clone)
        let carried = await clone?.vocabularyBias
        XCTAssertEqual(carried?.shape, .contextGraph)
        XCTAssertEqual(carried?.boost, 0.5)
        XCTAssertEqual(carried?.holdFrameOnShorterFlip, true)

        await manager.setVocabularyBias(nil)
        let plainClone = await manager.makeWorkerClone()
        let plainCarried = await plainClone?.vocabularyBias
        XCTAssertNil(plainCarried)
    }
}
