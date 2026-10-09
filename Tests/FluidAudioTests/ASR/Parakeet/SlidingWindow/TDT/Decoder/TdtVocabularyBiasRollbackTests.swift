// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5. It tests
// the vocabulary bias's completion rollback through `TdtDecoderV3.decodeWithTimings` with scripted
// models.

@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

private let blank = ScriptedTdt.blank

private func blankStep(_ durationBin: Int = 1) -> ScriptedJointStep {
    ScriptedJointStep(token: blank, probability: 0.9, durationBin: durationBin, topK: [(blank, 6), (8, 1)])
}

private func step(_ token: Int, _ durationBin: Int, _ topK: [(id: Int, logit: Float)]) -> ScriptedJointStep {
    ScriptedJointStep(token: token, probability: 0.8, durationBin: durationBin, topK: topK)
}

/// The decoder LSTM, scripted with a memory: its projection carries the token it was fed in element
/// 0, as `ScriptedDecoderModel`'s does, and it writes into the output backings a record of every
/// token fed along the state's lineage (element 0 the last token, 1 how many, 2 their sum, and the
/// cell's element 1 an order-sensitive hash). Two decodes end in equal arrays only if their LSTMs
/// were fed the same tokens in the same order from the same start.
private final class LineageDecoderModel: MLModel {
    private(set) var fedTokens: [Int] = []

    override func prediction(
        from input: MLFeatureProvider, options: MLPredictionOptions = MLPredictionOptions()
    ) throws -> MLFeatureProvider {
        guard let targets = input.featureValue(for: "targets")?.multiArrayValue,
            let hidden = input.featureValue(for: "h_in")?.multiArrayValue,
            let cell = input.featureValue(for: "c_in")?.multiArrayValue
        else {
            throw ASRError.processingFailed("lineage decoder: missing inputs")
        }
        let token = targets[0].intValue
        fedTokens.append(token)
        let count = hidden[1].floatValue + 1
        let sum = hidden[2].floatValue + Float(token)
        let hash = Float((Int(cell[1].floatValue) * 31 + token) % 65_521)
        let projection = try MLMultiArray(
            shape: [1, NSNumber(value: ASRConstants.decoderHiddenSize), 1], dataType: .float32)
        projection.resetData(to: 0)
        projection[0] = NSNumber(value: Float(token))
        var outputs: [String: MLFeatureValue] = ["decoder": MLFeatureValue(multiArray: projection)]
        if let h = options.outputBackings["h_out"] as? MLMultiArray {
            h[0] = NSNumber(value: Float(token))
            h[1] = NSNumber(value: count)
            h[2] = NSNumber(value: sum)
            outputs["h_out"] = MLFeatureValue(multiArray: h)
        }
        if let c = options.outputBackings["c_out"] as? MLMultiArray {
            c[0] = NSNumber(value: Float(token))
            c[1] = NSNumber(value: hash)
            outputs["c_out"] = MLFeatureValue(multiArray: c)
        }
        return try MLDictionaryFeatureProvider(dictionary: outputs)
    }
}

final class TdtVocabularyBiasRollbackTests: XCTestCase {

    // A toy piece table (a leading space marks a word start).
    private let vocab: [Int: String] = [
        8: " the", 9: " Ch", 10: " Si", 11: "ob", 12: "han", 13: "av", 20: " sit", 23: "ting", 26: "x",
        27: ".", 30: " D", 31: "ory", 32: "ori", 33: "an", 34: " floated", 40: " Old", 41: " New", 42: " York",
        43: " City", 44: " Jersey", 45: " Times", 46: " Square", 50: " \u{0414}", 51: " \u{0414}\u{0430}",
        52: " \u{0414}\u{0430}\u{043B}\u{0438}", 53: "\u{0440}\u{044C}\u{044F}", 8192: "<blank>",
    ]
    /// The piece table the script filter reads, with the blank's text empty as some vocabularies
    /// have it, so the filter may choose the blank.
    private var filterVocab: [Int: String] { vocab.merging([blank: ""]) { _, empty in empty } }
    private let siobhan = CustomVocabularyTerm(text: "Siobhan", tokenIds: [10, 11, 12])
    private let dorian = CustomVocabularyTerm(text: "Dorian", tokenIds: [30, 32, 33])
    private let chav = CustomVocabularyTerm(text: "Chav", tokenIds: [9, 13])
    private let newYorkCity = CustomVocabularyTerm(text: "New York City", tokenIds: [41, 42, 43])
    private let york = CustomVocabularyTerm(text: "York", tokenIds: [42])
    private let yorkTimes = CustomVocabularyTerm(text: "York Times", tokenIds: [42, 45])
    private let yorkTimesSquare = CustomVocabularyTerm(text: "York Times Square", tokenIds: [42, 45, 46])
    private let newYorkTimesSquare = CustomVocabularyTerm(text: "New York Times Square", tokenIds: [41, 42, 45, 46])

    private func bias(
        _ terms: [CustomVocabularyTerm]? = nil, rollback: Bool, minLetters: Int = 2
    ) -> TdtVocabularyBias {
        TdtVocabularyBias(
            terms: terms ?? [siobhan], vocabulary: vocab, blankId: blank, boost: 1, shape: .contextGraph,
            freshStartMinLetters: minLetters, holdFrameOnShorterFlip: true, completionRollback: rollback)!
    }

    private struct Decode {
        let hypothesis: TdtHypothesis
        let state: TdtDecoderState
        let joint: ScriptedJointModel
        let decoder: LineageDecoderModel
    }

    private func decode(
        frames: Int, bias: TdtVocabularyBias?, isLastChunk: Bool = false, emitAfter: Int? = nil,
        from start: TdtDecoderState? = nil, config: ASRConfig = ScriptedTdt.config, language: Language? = nil,
        script: @escaping (_ frame: Int, _ lastToken: Int) -> ScriptedJointStep
    ) async throws -> Decode {
        let joint = ScriptedJointModel(script: script)
        let decoder = LineageDecoderModel()
        var state = try start.map { try TdtDecoderState(from: $0) } ?? TdtDecoderState()
        let hypothesis = try await TdtDecoderV3(config: config).decodeWithTimings(
            encoderOutput: try ScriptedTdt.encoderOutput(frames: frames), encoderSequenceLength: frames,
            actualAudioFrames: frames, decoderModel: decoder, jointModel: joint, decoderState: &state,
            isLastChunk: isLastChunk, language: language, vocabulary: language == nil ? nil : filterVocab,
            emitTokensAfterGlobalFrame: emitAfter, vocabularyBias: bias)
        return Decode(hypothesis: hypothesis, state: state, joint: joint, decoder: decoder)
    }

    private func values(_ array: MLMultiArray?) -> [Float] {
        guard let array else { return [] }
        return (0..<array.count).map { array[$0].floatValue }
    }

    /// Every field of the hypotheses, and of the decoder states the decodes leave, are equal.
    private func assertSameDecode(_ a: Decode, _ b: Decode, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.hypothesis.ySequence, b.hypothesis.ySequence, "tokens", file: file, line: line)
        XCTAssertEqual(a.hypothesis.timestamps, b.hypothesis.timestamps, "timestamps", file: file, line: line)
        XCTAssertEqual(a.hypothesis.tokenDurations, b.hypothesis.tokenDurations, "durations", file: file, line: line)
        XCTAssertEqual(a.hypothesis.tokenConfidences, b.hypothesis.tokenConfidences, file: file, line: line)
        XCTAssertEqual(a.hypothesis.score, b.hypothesis.score, file: file, line: line)
        XCTAssertEqual(a.hypothesis.suppressedTokens, b.hypothesis.suppressedTokens, file: file, line: line)
        XCTAssertEqual(a.hypothesis.suppressedTimestamps, b.hypothesis.suppressedTimestamps, file: file, line: line)
        XCTAssertEqual(a.hypothesis.lastToken, b.hypothesis.lastToken, file: file, line: line)
        XCTAssertEqual(values(a.state.hiddenState), values(b.state.hiddenState), "LSTM h", file: file, line: line)
        XCTAssertEqual(values(a.state.cellState), values(b.state.cellState), "LSTM c", file: file, line: line)
        XCTAssertEqual(values(a.state.predictorOutput), values(b.state.predictorOutput), file: file, line: line)
        XCTAssertEqual(a.state.lastToken, b.state.lastToken, file: file, line: line)
        XCTAssertEqual(a.state.timeJump, b.state.timeJump, file: file, line: line)
    }

    /// " sit" flips to " Si" (Siobhan's opening, shorter, so the frame is held), then the audio goes
    /// on with "ting", not "ob".
    private func sitting(_ frame: Int, _ last: Int) -> ScriptedJointStep {
        switch (frame, last) {
        case (0, blank), (0, 27): return step(20, 1, [(20, 5), (10, 4.5), (blank, 1)])
        case (0, 10): return step(23, 1, [(23, 5), (11, 1)])
        case (1, 20): return step(23, 1, [(23, 5), (11, 1)])
        default: return blankStep()
        }
    }

    /// Without the rollback (the default) the abandoned opening stays.
    func testWithoutTheRollbackAFlipWhoseTermNeverCompletesStays() async throws {
        let shipped = TdtVocabularyBias(
            terms: [siobhan], vocabulary: vocab, blankId: blank, boost: 1, shape: .contextGraph,
            freshStartMinLetters: 2, holdFrameOnShorterFlip: true)!
        XCTAssertFalse(shipped.completionRollback)
        let decoded = try await decode(frames: 3, bias: shipped, script: sitting)
        XCTAssertEqual(decoded.hypothesis.ySequence, [10, 23], "\" Si\" + \"ting\"")
    }

    /// The match breaks at "ting": the decoder returns to the flip and takes the plain " sit", and the
    /// decode from there is the unbiased decode, frame for frame, its suppressed tokens included.
    func testAFlipWhoseTermNeverCompletesIsRolledBack() async throws {
        for emitAfter in [nil, 1] as [Int?] {
            let rolled = try await decode(frames: 3, bias: bias(rollback: true), emitAfter: emitAfter, script: sitting)
            let plain = try await decode(frames: 3, bias: nil, emitAfter: emitAfter, script: sitting)
            assertSameDecode(rolled, plain)
            XCTAssertEqual(rolled.hypothesis.ySequence, emitAfter == nil ? [20, 23] : [23])
            XCTAssertEqual(rolled.hypothesis.suppressedTokens, emitAfter == nil ? [] : [20])
            XCTAssertEqual(rolled.decoder.fedTokens, [blank, 10, 20, 23], "\" Si\" was fed once, then the replay")
            XCTAssertEqual(rolled.joint.frames, [0, 0] + plain.joint.frames, "no frame skipped or read twice")
        }
    }

    /// The step counters return with the rollback. The same-frame guard: the replayed " sit" may take
    /// duration 0 at frame 0, as the unbiased decode's does, though the abandoned " Si" was emitted
    /// there. The token budget: with two tokens a chunk, the replay still emits both.
    func testTheStepCountersAfterARollbackAreTheUnbiasedDecodes() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(20, 0, [(20, 5), (10, 4.5), (blank, 1)])
            case (0, 10), (0, 20): return step(23, 1, [(23, 5), (11, 1)])
            default: return blankStep()
            }
        }
        let rolled = try await decode(frames: 3, bias: bias(rollback: true), script: script)
        let plain = try await decode(frames: 3, bias: nil, script: script)
        XCTAssertEqual(plain.hypothesis.tokenDurations, [0, 1])
        assertSameDecode(rolled, plain)

        let twoTokens = ASRConfig(tdtConfig: TdtConfig(maxTokensPerChunk: 2), encoderHiddenSize: 1)
        let budgeted = try await decode(frames: 3, bias: bias(rollback: true), config: twoTokens, script: sitting)
        let budgetedPlain = try await decode(frames: 3, bias: nil, config: twoTokens, script: sitting)
        XCTAssertEqual(budgeted.hypothesis.ySequence, [20, 23])
        assertSameDecode(budgeted, budgetedPlain)
    }

    /// The token budget ends the chunk on a step that flipped: the flip is never emitted, yet its
    /// held frame would move where the chunk ends. The decode returns to that step and takes its
    /// plain argmax, so the chunk ends where the unbiased decode's does.
    func testAFlipOnTheStepTheTokenBudgetEndsIsUndone() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank), (1, 20): return step(20, 1, [(20, 5), (10, 4.5), (blank, 1)])
            case (0, 10), (1, 10): return step(23, 1, [(23, 5), (11, 1)])
            default: return blankStep()
            }
        }
        let oneToken = ASRConfig(tdtConfig: TdtConfig(maxTokensPerChunk: 1), encoderHiddenSize: 1)
        let rolled = try await decode(frames: 4, bias: bias(rollback: true), config: oneToken, script: script)
        let plain = try await decode(frames: 4, bias: nil, config: oneToken, script: script)
        XCTAssertEqual(plain.hypothesis.ySequence, [20])
        assertSameDecode(rolled, plain)
    }

    /// The same with a flip already waiting: " Jersey" flips to " York" toward New York City, then the
    /// budget ends the chunk on " Times" flipped to " City", which holds the frame. That step takes
    /// " Times" and moves on, so the last-chunk flush reads frame 2, where "x" breaks the match and
    /// " York" rolls back. Left in place, the held frame would have the flush read " Times" again and
    /// complete York Times.
    func testAFlipOnTheStepTheTokenBudgetEndsIsUndoneWhileAnotherWaits() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(41, 1, [(41, 5)])
            case (1, 41): return step(44, 1, [(44, 5), (42, 4.5)])
            case (1, 42): return step(45, 1, [(45, 5), (43, 4.5)])
            case (2, 42): return step(26, 1, [(26, 5)])
            default: return blankStep()
            }
        }
        let twoTokens = ASRConfig(tdtConfig: TdtConfig(maxTokensPerChunk: 2), encoderHiddenSize: 1)
        let terms = [newYorkCity, yorkTimes]
        let rolled = try await decode(
            frames: 5, bias: bias(terms, rollback: true), isLastChunk: true, config: twoTokens, script: script)
        let plain = try await decode(frames: 5, bias: nil, isLastChunk: true, config: twoTokens, script: script)
        XCTAssertEqual(plain.hypothesis.ySequence, [41, 44])
        assertSameDecode(rolled, plain)
    }

    /// A flip whose term completes stays, and nothing is replayed.
    func testAFlipWhoseTermCompletesStays() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(20, 1, [(20, 5), (10, 4.5), (blank, 1)])
            case (0, 10): return step(11, 1, [(11, 5), (23, 1)])
            case (1, 11): return step(12, 1, [(12, 5), (23, 1)])
            default: return blankStep()
            }
        }
        let kept = try await decode(frames: 3, bias: bias(rollback: true), script: script)
        let shipped = try await decode(frames: 3, bias: bias(rollback: false), script: script)
        XCTAssertEqual(kept.hypothesis.ySequence, [10, 11, 12])
        assertSameDecode(kept, shipped)
        XCTAssertEqual(kept.decoder.fedTokens, [blank, 10, 11, 12])
    }

    /// Two flips in one match (" sit" to " Si", then "x" to "ob"), then the match breaks: the
    /// decoder returns to the match's first flip, not its last.
    func testTheRestorePointIsTheMatchsFirstFlip() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(20, 1, [(20, 5), (10, 4.5), (blank, 1)])
            case (0, 10): return step(26, 1, [(26, 5), (11, 3.5)])
            case (1, 11), (1, 20): return step(23, 1, [(23, 5), (12, 0)])
            default: return blankStep()
            }
        }
        let shipped = try await decode(frames: 4, bias: bias(rollback: false), script: script)
        XCTAssertEqual(shipped.hypothesis.ySequence, [10, 11, 23], "both flips happen without the rollback")
        let rolled = try await decode(frames: 4, bias: bias(rollback: true), script: script)
        let plain = try await decode(frames: 4, bias: nil, script: script)
        XCTAssertEqual(rolled.hypothesis.ySequence, [20, 23])
        assertSameDecode(rolled, plain)
    }

    /// The decoder opens " D" itself and the bias flips "ory" to "ori" toward Dorian; the next piece
    /// is " floated": the continuation flip rolls back to "ory" ("The gamer Dory floated away.").
    /// Built with a one-letter guard, the app's setting.
    func testAContinuationFlipInAMatchTheDecoderOpenedRollsBack() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(30, 1, [(30, 5), (8, 1)])
            case (1, 30): return step(31, 1, [(31, 5), (32, 3.5)])
            case (2, 31), (2, 32): return step(34, 1, [(34, 5), (33, 0.5)])
            default: return blankStep()
            }
        }
        let shipped = try await decode(frames: 4, bias: bias([dorian], rollback: false, minLetters: 1), script: script)
        XCTAssertEqual(shipped.hypothesis.ySequence, [30, 32, 34], "\" D\" + \"ori\" + \" floated\"")
        let rolled = try await decode(frames: 4, bias: bias([dorian], rollback: true, minLetters: 1), script: script)
        let plain = try await decode(frames: 4, bias: nil, script: script)
        XCTAssertEqual(rolled.hypothesis.ySequence, [30, 31, 34])
        assertSameDecode(rolled, plain)
    }

    /// " Old" flips to " New", opening New York City; then " York" completes the term York, which
    /// does not hold the flip, and the graph leaves New York City for the root: the flip rolls back.
    func testATermCompletedAfterTheFlipThatDoesNotHoldItRollsTheFlipBack() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(40, 1, [(40, 5), (41, 4.5)])
            case (1, 40), (1, 41): return step(42, 1, [(42, 5)])
            default: return blankStep()
            }
        }
        let terms = [newYorkCity, york]
        let shipped = try await decode(frames: 3, bias: bias(terms, rollback: false), script: script)
        XCTAssertEqual(shipped.hypothesis.ySequence, [41, 42])
        let rolled = try await decode(frames: 3, bias: bias(terms, rollback: true), script: script)
        let plain = try await decode(frames: 3, bias: nil, script: script)
        XCTAssertEqual(rolled.hypothesis.ySequence, [40, 42])
        assertSameDecode(rolled, plain)
    }

    /// After " New", " Jersey" flips to " York" toward New York City; then " Times" leaves that match
    /// through the failure link for one that still holds the flip. York Times completes there, and
    /// York Times Square one token later: either way the flip stays.
    func testATermCompletedThroughTheFailureLinkThatHoldsTheFlipKeepsIt() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(41, 1, [(41, 5)])
            case (1, 41): return step(44, 1, [(44, 5), (42, 4.5)])
            case (1, 42): return step(45, 1, [(45, 5)])
            case (2, 45): return step(46, 1, [(46, 5)])
            default: return blankStep()
            }
        }
        for terms in [[newYorkCity, yorkTimes], [newYorkCity, yorkTimesSquare]] {
            let kept = try await decode(frames: 4, bias: bias(terms, rollback: true), script: script)
            let shipped = try await decode(frames: 4, bias: bias(terms, rollback: false), script: script)
            XCTAssertEqual(kept.hypothesis.ySequence, [41, 42, 45, 46])
            assertSameDecode(kept, shipped)
            XCTAssertEqual(kept.decoder.fedTokens, shipped.decoder.fedTokens, "nothing replayed")
        }
    }

    /// A shorter term completed through the output link inside a longer match holds the flip: after
    /// " New", " Jersey" flips to " York". With New York City and York listed, the flip completes York
    /// by itself; with New York Times Square and York Times listed, " Times" completes York Times.
    /// Either way the flip stays.
    func testATermCompletedThroughTheOutputLinkThatHoldsTheFlipKeepsIt() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(41, 1, [(41, 5)])
            case (1, 41): return step(44, 1, [(44, 5), (42, 4.5)])
            case (1, 42): return step(45, 1, [(45, 5)])
            default: return blankStep()
            }
        }
        for terms in [[newYorkCity, york], [newYorkTimesSquare, yorkTimes]] {
            let kept = try await decode(frames: 4, bias: bias(terms, rollback: true), script: script)
            let shipped = try await decode(frames: 4, bias: bias(terms, rollback: false), script: script)
            XCTAssertEqual(kept.hypothesis.ySequence, [41, 42, 45])
            assertSameDecode(kept, shipped)
            XCTAssertEqual(kept.decoder.fedTokens, shipped.decoder.fedTokens, "nothing replayed")
        }
    }

    /// A flip that completes a term by itself (" Jersey" to " York", with York listed) stays, though
    /// what follows continues no term.
    func testAFlipThatCompletesATermByItselfStays() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(44, 1, [(44, 5), (42, 4.5)])
            case (1, 42), (1, 44): return step(45, 1, [(45, 5)])
            default: return blankStep()
            }
        }
        let kept = try await decode(frames: 3, bias: bias([york], rollback: true), script: script)
        let shipped = try await decode(frames: 3, bias: bias([york], rollback: false), script: script)
        XCTAssertEqual(kept.hypothesis.ySequence, [42, 45])
        assertSameDecode(kept, shipped)
        XCTAssertEqual(kept.decoder.fedTokens, shipped.decoder.fedTokens, "nothing replayed")
    }

    /// A flip still waiting when the chunk's audio ends never completed: it rolls back.
    func testAFlipStillWaitingWhenTheAudioEndsRollsBack() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(20, 1, [(20, 5), (10, 4.5), (blank, 1)])
            case (0, 10): return blankStep(2)
            default: return blankStep()
            }
        }
        let shipped = try await decode(frames: 2, bias: bias(rollback: false), script: script)
        XCTAssertEqual(shipped.hypothesis.ySequence, [10])
        let rolled = try await decode(frames: 2, bias: bias(rollback: true), script: script)
        let plain = try await decode(frames: 2, bias: nil, script: script)
        XCTAssertEqual(rolled.hypothesis.ySequence, [20])
        assertSameDecode(rolled, plain)
    }

    /// The flip lands in the inner blank loop at frame 1; the main loop then ends, and the last-chunk
    /// flush decodes what the term needs: "ob" and "han" complete Siobhan, so the flip stays.
    func testATermTheLastChunkFlushCompletesStays() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (1, blank): return step(20, 1, [(20, 5), (10, 4.5), (blank, 1)])
            case (1, 10): return blankStep(2)
            case (2, 10): return step(11, 1, [(11, 5)])
            case (2, 11): return step(12, 1, [(12, 5)])
            default: return blankStep()
            }
        }
        let kept = try await decode(frames: 3, bias: bias(rollback: true), isLastChunk: true, script: script)
        let shipped = try await decode(frames: 3, bias: bias(rollback: false), isLastChunk: true, script: script)
        XCTAssertEqual(kept.hypothesis.ySequence, [10, 11, 12])
        assertSameDecode(kept, shipped)
        XCTAssertEqual(kept.decoder.fedTokens, shipped.decoder.fedTokens, "nothing replayed")
        XCTAssertEqual(kept.joint.frames, shipped.joint.frames)
    }

    /// The same flip, and the flush decodes "x", which breaks the match: the decoder returns to the
    /// flip at once, before the LSTM reads "x" or the flush decodes " Ch" + "av", a term that does
    /// not hold the flip.
    func testAMatchTheLastChunkFlushBreaksRollsBackAtOnce() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (1, blank): return step(20, 1, [(20, 5), (10, 4.5), (blank, 1)])
            case (1, 10): return blankStep(2)
            case (2, 10): return step(26, 1, [(26, 5)])
            case (2, 26): return step(9, 1, [(9, 5)])
            case (1, 9): return step(13, 1, [(13, 5)])
            default: return blankStep()
            }
        }
        let terms = [siobhan, chav]
        let shipped = try await decode(frames: 3, bias: bias(terms, rollback: false), isLastChunk: true, script: script)
        XCTAssertEqual(shipped.hypothesis.ySequence, [10, 26, 9, 13])
        let rolled = try await decode(frames: 3, bias: bias(terms, rollback: true), isLastChunk: true, script: script)
        let plain = try await decode(frames: 3, bias: nil, isLastChunk: true, script: script)
        XCTAssertEqual(rolled.hypothesis.ySequence, [20])
        assertSameDecode(rolled, plain)
        XCTAssertEqual(rolled.decoder.fedTokens, [blank, 10, 20])
    }

    /// With a language set the rollback is off and the decode is the shipped bias's: the script
    /// filter runs after the bias and can replace a flip (here a Cyrillic argmax the bias leaves alone
    /// becomes a blank, and the inner blank loop decides again; or a flip itself becomes a blank and
    /// still holds the frame), which the rollback does not model.
    func testWithALanguageSetTheRollbackIsOff() async throws {
        let decidesTwice: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(50, 1, [(50, 5), (blank, 4.8), (10, 4.5)])
            case (1, blank): return step(50, 1, [(50, 5), (blank, 4.8), (9, 4.5)])
            case (1, 10), (2, 9): return step(26, 1, [(26, 5)])
            default: return blankStep()
            }
        }
        let flipFilteredAway: (Int, Int) -> ScriptedJointStep = { frame, last in
            switch (frame, last) {
            case (0, blank): return step(52, 3, [(52, 5), (blank, 4.8), (51, 4.5)])
            case (1, blank): return step(20, 1, [(20, 5), (10, 4.5)])
            case (1, 10): return step(26, 1, [(26, 5)])
            default: return blankStep()
            }
        }
        let terms = [siobhan, chav, CustomVocabularyTerm(text: "Darya", tokenIds: [51, 53])]
        for script in [decidesTwice, flipFilteredAway] {
            let watched = try await decode(
                frames: 5, bias: bias(terms, rollback: true), language: .english, script: script)
            let shipped = try await decode(
                frames: 5, bias: bias(terms, rollback: false), language: .english, script: script)
            assertSameDecode(watched, shipped)
            XCTAssertEqual(watched.decoder.fedTokens, shipped.decoder.fedTokens, "nothing replayed")
        }
    }

    /// After a rollback the decoder's state is the unbiased decode's: the LSTM arrays hold the same
    /// lineage and are still the caller's arrays, as the decoder writes them in place. Also from a
    /// state whose first step runs the LSTM before the flip (a chunk after one that ended on
    /// punctuation: last token ".", no cached projection).
    func testTheDecoderStateAfterARollbackIsTheUnbiasedDecodesState() async throws {
        var fresh = try TdtDecoderState()
        fresh.hiddenState[5] = 0.25
        var afterPunctuation = try TdtDecoderState()
        afterPunctuation.lastToken = 27
        afterPunctuation.hiddenState[1] = 3
        afterPunctuation.cellState[1] = 7
        for start in [fresh, afterPunctuation] {
            var state = try TdtDecoderState(from: start)
            let hidden = state.hiddenState
            let cell = state.cellState
            let joint = ScriptedJointModel(script: sitting)
            let decoder = LineageDecoderModel()
            let hypothesis = try await TdtDecoderV3(config: ScriptedTdt.config).decodeWithTimings(
                encoderOutput: try ScriptedTdt.encoderOutput(frames: 3), encoderSequenceLength: 3,
                actualAudioFrames: 3, decoderModel: decoder, jointModel: joint, decoderState: &state,
                vocabularyBias: bias(rollback: true))
            let rolled = Decode(hypothesis: hypothesis, state: state, joint: joint, decoder: decoder)
            let plain = try await decode(frames: 3, bias: nil, from: start, script: sitting)
            XCTAssertEqual(rolled.hypothesis.ySequence, [20, 23])
            assertSameDecode(rolled, plain)
            XCTAssertTrue(state.hiddenState === hidden && state.cellState === cell, "the caller's arrays")
        }
    }

    /// A decode keeps at most 64 rollback points, so its replays are bounded; past the 64th rollback
    /// the bias flips no more, so no flip stays unwatched and the rest decodes as with no bias.
    func testPastSixtyFourRollbacksTheBiasFlipsNoMore() async throws {
        let script: (Int, Int) -> ScriptedJointStep = { _, last in
            last == 10 ? step(23, 1, [(23, 5), (11, 1)]) : step(20, 1, [(20, 5), (10, 4.5), (blank, 1)])
        }
        let capped = try await decode(frames: 66, bias: bias(rollback: true), script: script)
        let plain = try await decode(frames: 66, bias: nil, script: script)
        assertSameDecode(capped, plain)
        XCTAssertEqual(capped.decoder.fedTokens.filter { $0 == 10 }.count, 64, "64 flips rolled back, then none")
    }

    /// The manager builds the rollback into the bias, and its chunk workers carry it.
    func testTheManagerBuildsTheRollbackAndItsWorkersCarryIt() async throws {
        let models = AsrModels(
            encoder: nil, preprocessor: ScriptedDecoderModel(), decoder: ScriptedDecoderModel(),
            joint: ScriptedDecoderModel(), configuration: MLModelConfiguration(), vocabulary: vocab, version: .ultra)
        let manager = AsrManager(config: .default)
        try await manager.loadModels(models)
        let built = await manager.makeVocabularyBias(
            terms: [siobhan], boost: 0.5, shape: .contextGraph, freshStartMinLetters: 1, holdFrameOnShorterFlip: true,
            completionRollback: true)
        XCTAssertEqual(built?.completionRollback, true)
        XCTAssertEqual(built?.freshStartMinLetters, 1)
        await manager.setVocabularyBias(built)
        let clone = await manager.makeWorkerClone()
        let carried = await clone?.vocabularyBias
        XCTAssertEqual(carried?.completionRollback, true)
        let plain = await manager.makeVocabularyBias(terms: [siobhan], boost: 0.5, shape: .contextGraph)
        XCTAssertEqual(plain?.completionRollback, false, "off unless asked for")
    }
}
