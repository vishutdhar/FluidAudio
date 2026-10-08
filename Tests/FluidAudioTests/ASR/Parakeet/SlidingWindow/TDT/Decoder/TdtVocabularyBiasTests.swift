import XCTest

@testable import FluidAudio

final class TdtVocabularyBiasTests: XCTestCase {

    // A toy piece table in the TDT vocabulary's format (a leading space marks a word start).
    private let vocab: [Int: String] = [
        1: " S", 2: "up", 3: "ab", 4: "ase", 5: " Su", 6: "per", 7: "b", 8: " the", 9: " Ch",
        10: " Si", 11: "ob", 12: "han", 13: "av", 14: "on", 15: "pa", 16: " Ter", 17: "min", 18: " termin",
        19: "al", 20: "als", 8192: "<blank>",
    ]
    private let blankId = 8192
    private let siobhan = CustomVocabularyTerm(text: "Siobhan", tokenIds: [10, 11, 12])
    private let supabase = CustomVocabularyTerm(text: "Supabase", tokenIds: [1, 2, 3, 4])

    private func bias(
        _ terms: [CustomVocabularyTerm], _ shape: TdtVocabularyBias.Shape, boost: Float = 1,
        minLetters: Int = 2, hold: Bool = false
    ) -> TdtVocabularyBias {
        TdtVocabularyBias(
            terms: terms, vocabulary: vocab, blankId: blankId, boost: boost, shape: shape,
            freshStartMinLetters: minLetters, holdFrameOnShorterFlip: hold)!
    }

    func testBlankArgmaxIsNeverOverridden() {
        for shape in TdtVocabularyBias.Shape.allCases {
            let b = bias([siobhan, supabase], shape, boost: 6, minLetters: 0)
            var state = b.makeState()
            let picked = b.select(plain: blankId, topKIds: [blankId, 10, 5], topKLogits: [5, 4.9, 4.9], state: &state)
            XCTAssertNil(picked, "\(shape)")
        }
    }

    func testNoUsableTermsBuildsNothing() {
        let short = CustomVocabularyTerm(text: "ab", tokenIds: nil)
        XCTAssertNil(TdtVocabularyBias(terms: [short], vocabulary: vocab, blankId: blankId, boost: 1, shape: .pieceTrie))
        XCTAssertNil(TdtVocabularyBias(terms: [short], vocabulary: vocab, blankId: blankId, boost: 1, shape: .contextGraph))
        XCTAssertNil(TdtVocabularyBias(terms: [siobhan], vocabulary: vocab, blankId: blankId, boost: 0, shape: .pieceTrie))
    }

    func testPieceTrieOpensOnlyWithTwoLetters() {
        let b = bias([supabase], .pieceTrie)
        var state = b.makeState()
        // " S" is one letter: not boosted. " Su" is: 4.5 + 1 beats 5.
        let picked = b.select(plain: 8, topKIds: [8, 1, 5], topKLogits: [5, 4.95, 4.5], state: &state)
        XCTAssertEqual(picked?.tokenId, 5)
        XCTAssertNil(b.select(plain: 8, topKIds: [8, 1, 5], topKLogits: [5, 4.95, 3.9], state: &state))
    }

    func testPieceTrieFollowsTheDecodersSegmentation() {
        let b = bias([supabase], .pieceTrie)
        var state = b.makeState()
        b.observe(5, state: &state)  // " Su", not in the fixed encoding " S up ab ase"
        let picked = b.select(plain: 6, topKIds: [6, 15], topKLogits: [5, 4.5], state: &state)
        XCTAssertEqual(picked?.tokenId, 15)  // "pa" continues "▁supabase"; "per" does not
    }

    func testContextGraphBoostsArcsTwiceAndPaysBack() {
        let b = bias([siobhan], .contextGraph)
        var state = b.makeState()
        // Root: " Si" earns the in-place bonus and the arc score, 3.5 + 2 > 5.
        XCTAssertEqual(b.select(plain: 9, topKIds: [9, 10], topKLogits: [5, 3.5], state: &state)?.tokenId, 10)
        b.observe(10, state: &state)
        // In a match of score 1: "ob" earns 2, the plain "av" pays 1 back. 2.5 + 2 > 5 - 1.
        XCTAssertEqual(b.select(plain: 13, topKIds: [13, 11], topKLogits: [5, 2.5], state: &state)?.tokenId, 11)
        XCTAssertNil(b.select(plain: 13, topKIds: [13, 11], topKLogits: [5, 1.9], state: &state))
        b.observe(11, state: &state)
        b.observe(12, state: &state)
        // Completed: back at the root, nothing to pay back, nothing engaged by " the".
        XCTAssertNil(b.select(plain: 8, topKIds: [8, 14], topKLogits: [5, 4.9], state: &state))
    }

    func testContextGraphFreshStartGuard() {
        let guarded = bias([supabase], .contextGraph)
        var state = guarded.makeState()
        XCTAssertNil(guarded.select(plain: 8, topKIds: [8, 1], topKLogits: [5, 4.5], state: &state))
        // Still tracked: once the decoder itself says " S", "up" is boosted.
        guarded.observe(1, state: &state)
        XCTAssertEqual(guarded.select(plain: 6, topKIds: [6, 2], topKLogits: [5, 3.5], state: &state)?.tokenId, 2)

        let open = bias([supabase], .contextGraph, minLetters: 0)
        var openState = open.makeState()
        XCTAssertEqual(open.select(plain: 8, topKIds: [8, 1], topKLogits: [5, 4.5], state: &openState)?.tokenId, 1)
    }

    func testPieceGraphKeepsTheDecodersConsistentPiece() {
        let terminal = CustomVocabularyTerm(text: "Terminal", tokenIds: [16, 17, 19])
        // The fixed encoding is " Ter min al". The decoder says " termin", which spells the same
        // word: the piece graph boosts it like " Ter" and keeps it; the token graph flips it.
        let pieceGraph = bias([terminal], .pieceGraph)
        var s1 = pieceGraph.makeState()
        XCTAssertNil(pieceGraph.select(plain: 18, topKIds: [18, 16], topKLogits: [5, 4.5], state: &s1))
        let tokenGraph = bias([terminal], .contextGraph)
        var s2 = tokenGraph.makeState()
        XCTAssertEqual(tokenGraph.select(plain: 18, topKIds: [18, 16], topKLogits: [5, 4.5], state: &s2)?.tokenId, 16)
        // After " Ter", "min" continues and "als" pays the accumulated bonus back.
        pieceGraph.observe(16, state: &s1)
        XCTAssertEqual(pieceGraph.select(plain: 20, topKIds: [20, 17], topKLogits: [5, 2.5], state: &s1)?.tokenId, 17)
    }

    func testHoldFrameOnlyOnShorterFlips() {
        let term = CustomVocabularyTerm(text: "Freedom Terminal", tokenIds: [16, 17, 19])
        let held = bias([term], .contextGraph, hold: true)
        var state = held.makeState()
        var label = 18  // " termin" (6 letters), flipped to " Ter" (3 letters)
        var score: Float = 0.9
        XCTAssertTrue(
            held.apply(label: &label, score: &score, topKIds: [18, 16], topKLogits: [5, 3.5], state: &state))
        XCTAssertEqual(label, 16)
        var other = 9  // " Ch" (2 letters) to " Si" (2 letters): no hold
        let siobhanBias = bias([siobhan], .contextGraph, hold: true)
        var s2 = siobhanBias.makeState()
        XCTAssertFalse(
            siobhanBias.apply(label: &other, score: &score, topKIds: [9, 10], topKLogits: [5, 3.5], state: &s2))
        XCTAssertEqual(other, 10)
    }
}
