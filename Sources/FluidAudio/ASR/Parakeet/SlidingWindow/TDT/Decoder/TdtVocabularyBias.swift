// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5: the decode-time
// vocabulary bias for the TDT v3 greedy decoder.

import Foundation

/// Decode-time custom vocabulary biasing for the TDT v3 greedy decoder.
///
/// The v3 joint already returns its 64 highest token logits (`top_k_ids`, `top_k_logits`) for the
/// script filter. This biases the greedy choice among them at the same two sites: when the plain
/// argmax is a non-blank token, a candidate that continues a listed word may replace it if its
/// logit plus the bonus beats the plain token's (boosted when it is itself a candidate). A blank
/// argmax is never overridden, so silence never fills with words, and the duration bin is always
/// the model's own.
///
/// The index is immutable and `Sendable` (chunks decode in parallel); the per-decode match state
/// is a small value the decoder owns for one `decodeWithTimings` call.
///
/// Two match shapes:
/// - `pieceTrie`: `NemotronVocabularyBias`'s matcher on the TDT piece table. Matching is on piece
///   text, so any segmentation of the word the decoder drifts toward stays boostable. Every vocab
///   piece that continues a listed word from any viable offset gets a flat bonus; opening a fresh
///   word needs a piece of at least two letters.
/// - `contextGraph`: a port of sherpa-onnx's `ContextGraph` (v1.13.4) as its NeMo modified beam
///   search uses it, reduced to one greedy path: an Aho-Corasick graph over each word's fixed token
///   ids (`CustomVocabularyTerm.tokenIds`), the bonus added to every arc out of the current state
///   and again as the arc's graph score, failure links, and the accumulated score paid back when a
///   partial match breaks.
/// - `pieceGraph`: sherpa's scoring on `pieceTrie`'s piece-text matching. A token that continues
///   the deepest live match earns the bonus twice (the in-place bonus and the graph score); a token
///   that continues a shorter live match or opens a word earns the graph score less what the
///   deepest match has accumulated; a token that continues nothing pays the accumulated score
///   back. The accumulated score is the bonus times the tokens the deepest match has consumed.
public struct TdtVocabularyBias: Sendable {

    public enum Shape: String, Sendable, CaseIterable {
        case pieceTrie
        case contextGraph
        case pieceGraph
    }

    /// Per-decode match state.
    public struct State: Sendable {
        var tail: [UInt32] = []
        /// Tail offsets where each committed token starts (pieceGraph token counting).
        var tokenStarts: [Int] = []
        var cached: [Int: Float]? = nil
        /// pieceGraph: the payback a token that continues nothing would pay.
        var cachedPayback: Float = 0
        var graphNode: Int = 0
        public init() {}
    }

    public let shape: Shape
    public let boost: Float
    /// Fresh-word guard: a word may be opened only by a piece with at least this many letters
    /// (the shipped Nemotron matcher uses 2; a sherpa graph has no such guard).
    public let freshStartMinLetters: Int
    /// When a flip emits a piece that spells fewer letters than the plain argmax, stay on the
    /// frame (duration 0) instead of jumping the plain token's duration, so the audio the shorter
    /// piece did not spell is still decoded.
    public let holdFrameOnShorterFlip: Bool
    let blankId: Int
    private let lettersById: [Int: Int]
    private let trie: PieceTrieIndex?
    private let graph: ContextGraphIndex?

    static let marker: UInt32 = 0x2581

    /// Builds the bias, or `nil` when no usable term remains.
    /// - Parameters:
    ///   - terms: listed words. `contextGraph` reads `tokenIds` and skips terms without them.
    ///   - vocabulary: the loaded TDT id to piece table (`▁` or a leading space marks a word start).
    public init?(
        terms: [CustomVocabularyTerm],
        vocabulary: [Int: String],
        blankId: Int,
        boost: Float,
        shape: Shape,
        freshStartMinLetters: Int = 2,
        holdFrameOnShorterFlip: Bool = false
    ) {
        guard boost > 0 else { return nil }
        self.holdFrameOnShorterFlip = holdFrameOnShorterFlip
        self.shape = shape
        self.boost = boost
        self.blankId = blankId
        self.freshStartMinLetters = freshStartMinLetters
        let pieces = PieceTable(vocabulary: vocabulary)
        self.lettersById = pieces.scalarsById.mapValues(PieceTable.letters)
        switch shape {
        case .pieceTrie, .pieceGraph:
            guard
                let index = PieceTrieIndex(
                    terms: terms, pieces: pieces, boost: boost, freshStartMinLetters: freshStartMinLetters)
            else { return nil }
            trie = index
            graph = nil
        case .contextGraph:
            guard
                let index = ContextGraphIndex(
                    terms: terms, pieces: pieces, boost: boost, freshStartMinLetters: freshStartMinLetters)
            else { return nil }
            graph = index
            trie = nil
        }
    }

    public func makeState() -> State { State() }

    /// Record a committed non-blank token (emitted or suppressed: the LSTM sees both).
    public func observe(_ tokenId: Int, state: inout State) {
        if let trie {
            trie.observe(tokenId, state: &state)
        } else if let graph {
            state.graphNode = graph.forward(from: state.graphNode, token: tokenId).next
        }
    }

    /// The biased choice at one decode step, or `nil` to keep the plain argmax.
    /// - Returns: the picked id and its raw (unboosted) logit.
    public func select(
        plain: Int, topKIds: [Int], topKLogits: [Float], state: inout State
    ) -> (tokenId: Int, logit: Float)? {
        guard plain != blankId else { return nil }
        let count = min(topKIds.count, topKLogits.count)
        guard let plainIndex = topKIds[0..<count].firstIndex(of: plain) else { return nil }
        let plainLogit = topKLogits[plainIndex]
        var bestId = plain
        var bestLogit = plainLogit
        if shape == .pieceGraph, let trie {
            let deltas = trie.graphDeltas(state: &state, boost: boost)
            guard !deltas.isEmpty else { return nil }
            var bestScore = plainLogit + (deltas[plain] ?? -state.cachedPayback)
            for i in 0..<count {
                let id = topKIds[i]
                guard id != blankId, id != plain, let delta = deltas[id] else { continue }
                let score = topKLogits[i] + delta
                if score > bestScore {
                    bestScore = score
                    bestId = id
                    bestLogit = topKLogits[i]
                }
            }
        } else if let trie {
            let candidates = trie.candidates(state: &state)
            guard !candidates.isEmpty else { return nil }
            var bestScore = plainLogit
            for i in 0..<count {
                let id = topKIds[i]
                guard id != blankId, let bonus = candidates[id] else { continue }
                let score = topKLogits[i] + bonus
                if score > bestScore {
                    bestScore = score
                    bestId = id
                    bestLogit = topKLogits[i]
                }
            }
        } else if let graph {
            let node = state.graphNode
            var bestScore = plainLogit + graph.stepScore(from: node, token: plain).score
            for i in 0..<count {
                let id = topKIds[i]
                guard id != blankId, id != plain else { continue }
                let step = graph.stepScore(from: node, token: id)
                guard step.engaged else { continue }
                let score = topKLogits[i] + step.score
                if score > bestScore {
                    bestScore = score
                    bestId = id
                    bestLogit = topKLogits[i]
                }
            }
        }
        return bestId == plain ? nil : (bestId, bestLogit)
    }

    /// Applies `select` to a decode step, recomputing the score of a flipped token as its top-K
    /// softmax (the convention `applyEnglishBlocklist` uses).
    /// - Returns: true when the decoder should hold the frame (`holdFrameOnShorterFlip`).
    @discardableResult
    func apply(
        label: inout Int, score: inout Float, topKIds: [Int], topKLogits: [Float], state: inout State
    ) -> Bool {
        guard let picked = select(plain: label, topKIds: topKIds, topKLogits: topKLogits, state: &state) else {
            return false
        }
        let hold =
            holdFrameOnShorterFlip
            && (lettersById[picked.tokenId] ?? 0) < (lettersById[label] ?? 0)
        if tdtBiasLogEnabled {
            FileHandle.standardError.write(Data("tdt-bias-flip: \(label) -> \(picked.tokenId)\n".utf8))
        }
        label = picked.tokenId
        var maxLogit: Float = -.infinity
        for l in topKLogits where l > maxLogit { maxLogit = l }
        var sumExp: Float = 0
        for l in topKLogits { sumExp += expf(l - maxLogit) }
        score = sumExp > 0 ? expf(picked.logit - maxLogit) / sumExp : 0
        return hold
    }
}

/// `FLUIDAUDIO_TDT_BIAS_LOG=1` traces every flip on stderr.
let tdtBiasLogEnabled: Bool = {
    let value = ProcessInfo.processInfo.environment["FLUIDAUDIO_TDT_BIAS_LOG"] ?? ""
    return !(value.isEmpty || value == "0" || value.lowercased() == "false")
}()

/// The TDT piece table folded for matching: lowercased NFC, word-start marker as `▁`.
struct PieceTable: Sendable {
    /// Folded piece scalars per id. Special pieces (`<unk>`, `<|...|>`) are absent.
    let scalarsById: [Int: [UInt32]]
    /// Folded piece text to every id that folds to it (`▁su` to the ids of `▁Su`, `▁su`, `▁SU`).
    let idsByPiece: [String: [Int]]

    init(vocabulary: [Int: String]) {
        var scalarsById: [Int: [UInt32]] = [:]
        var idsByPiece: [String: [Int]] = [:]
        for (id, raw) in vocabulary {
            if raw.hasPrefix("<") && raw.hasSuffix(">") { continue }
            let folded = Self.fold(raw)
            guard !folded.isEmpty else { continue }
            scalarsById[id] = folded.unicodeScalars.map(\.value)
            idsByPiece[folded, default: []].append(id)
        }
        self.scalarsById = scalarsById
        self.idsByPiece = idsByPiece
    }

    static func fold(_ piece: String) -> String {
        piece.replacingOccurrences(of: " ", with: "\u{2581}").lowercased().precomposedStringWithCanonicalMapping
    }

    static func letters(_ folded: [UInt32]) -> Int {
        folded.filter { $0 != TdtVocabularyBias.marker }.count
    }
}

/// `NemotronVocabularyBias`'s matcher, as immutable arrays.
struct PieceTrieIndex: Sendable {
    struct Entry: Sendable {
        let form: [UInt32]
        let boost: Float
    }
    struct Node: Sendable {
        var children: [UInt32: Int] = [:]
        var entryIndices: [Int] = []
    }

    static let minTermLength = 3

    let entries: [Entry]
    let nodes: [Node]
    let freshStart: [Int: Float]
    let pieces: PieceTable
    let tailCap: Int
    let freshStartMinLetters: Int

    init?(terms: [CustomVocabularyTerm], pieces: PieceTable, boost: Float, freshStartMinLetters: Int) {
        var entries: [Entry] = []
        for term in terms {
            for surface in [term.text] + (term.aliases ?? []) {
                guard surface.filter({ !$0.isWhitespace }).count >= Self.minTermLength else { continue }
                entries.append(Entry(form: Self.pieceForm(surface), boost: boost))
            }
        }
        guard !entries.isEmpty else { return nil }
        var nodes = [Node()]
        for (index, entry) in entries.enumerated() {
            var node = 0
            for scalar in entry.form {
                if let next = nodes[node].children[scalar] {
                    node = next
                } else {
                    nodes.append(Node())
                    nodes[node].children[scalar] = nodes.count - 1
                    node = nodes.count - 1
                }
                nodes[node].entryIndices.append(index)
            }
        }
        var fresh: [Int: Float] = [:]
        for entry in entries {
            Self.accumulate(
                entry, offset: 0, pieces: pieces, freshStartMinLetters: freshStartMinLetters, into: &fresh)
        }
        self.entries = entries
        self.nodes = nodes
        self.freshStart = fresh
        self.pieces = pieces
        self.tailCap = entries.map(\.form.count).max() ?? 0
        self.freshStartMinLetters = freshStartMinLetters
    }

    /// `"Freedom Terminal"` to the scalars of `"▁freedom▁terminal"`.
    static func pieceForm(_ text: String) -> [UInt32] {
        let folded = text.lowercased().precomposedStringWithCanonicalMapping
        var form: [UInt32] = []
        var atBoundary = true
        for ch in folded {
            if ch.isWhitespace {
                atBoundary = true
                continue
            }
            if atBoundary {
                form.append(TdtVocabularyBias.marker)
                atBoundary = false
            }
            form.append(contentsOf: ch.unicodeScalars.map(\.value))
        }
        return form
    }

    func observe(_ tokenId: Int, state: inout TdtVocabularyBias.State) {
        guard let scalars = pieces.scalarsById[tokenId] else { return }
        state.tokenStarts.append(state.tail.count)
        state.tail.append(contentsOf: scalars)
        if state.tail.count > tailCap {
            let drop = state.tail.count - tailCap
            state.tail.removeFirst(drop)
            state.tokenStarts = state.tokenStarts.map { $0 - drop }.filter { $0 >= 0 }
        }
        state.cached = nil
    }

    /// Live strict-prefix matches at the current tail: (entry, offset), one per entry and offset.
    func liveMatches(_ tail: [UInt32]) -> [(entry: Int, offset: Int)] {
        var out: [(entry: Int, offset: Int)] = []
        guard !tail.isEmpty else { return out }
        for start in max(0, tail.count - tailCap)..<tail.count {
            var node = 0
            var matched = true
            for i in start..<tail.count {
                guard let next = nodes[node].children[tail[i]] else {
                    matched = false
                    break
                }
                node = next
            }
            guard matched else { continue }
            let offset = tail.count - start
            for index in nodes[node].entryIndices where offset < entries[index].form.count {
                out.append((index, offset))
            }
        }
        return out
    }

    /// pieceGraph: the score sherpa's beam search would add to a path for each boostable token,
    /// relative to the deepest live match. Tokens absent from the map pay `cachedPayback` back.
    func graphDeltas(state: inout TdtVocabularyBias.State, boost: Float) -> [Int: Float] {
        if let cached = state.cached { return cached }
        let live = liveMatches(state.tail)
        func consumed(_ offset: Int) -> Int {
            let start = state.tail.count - offset
            return state.tokenStarts.filter { $0 >= start }.count
        }
        let deepest = live.max { $0.offset < $1.offset }
        let accumulated = deepest.map { Float(consumed($0.offset)) * boost } ?? 0
        var deltas: [Int: Float] = [:]
        func merge(_ ids: [Int: Float], _ delta: Float) {
            for id in ids.keys where delta > deltas[id, default: -.infinity] { deltas[id] = delta }
        }
        for match in live {
            var ids: [Int: Float] = [:]
            Self.accumulate(
                entries[match.entry], offset: match.offset, pieces: pieces,
                freshStartMinLetters: freshStartMinLetters, into: &ids)
            let isDeepest = deepest.map { $0.entry == match.entry && $0.offset == match.offset } ?? false
            let delta = (isDeepest ? boost : 0) + Float(consumed(match.offset) + 1) * boost - accumulated
            merge(ids, delta)
        }
        merge(freshStart, (deepest == nil ? boost : 0) + boost - accumulated)
        state.cached = deltas
        state.cachedPayback = accumulated
        return deltas
    }

    /// Every vocab piece that continues a listed word from the fresh start or from any strict
    /// prefix of a word the emitted tail ends with.
    func candidates(state: inout TdtVocabularyBias.State) -> [Int: Float] {
        if let cached = state.cached { return cached }
        var best = freshStart
        let tail = state.tail
        if !tail.isEmpty {
            for start in max(0, tail.count - tailCap)..<tail.count {
                var node = 0
                var matched = true
                for i in start..<tail.count {
                    guard let next = nodes[node].children[tail[i]] else {
                        matched = false
                        break
                    }
                    node = next
                }
                guard matched else { continue }
                let offset = tail.count - start
                for index in nodes[node].entryIndices {
                    Self.accumulate(
                        entries[index], offset: offset, pieces: pieces,
                        freshStartMinLetters: freshStartMinLetters, into: &best)
                }
            }
        }
        state.cached = best
        return best
    }

    static func accumulate(
        _ entry: Entry, offset: Int, pieces: PieceTable, freshStartMinLetters: Int, into best: inout [Int: Float]
    ) {
        let form = entry.form
        guard offset < form.count else { return }
        var piece = String.UnicodeScalarView()
        var letters = 0
        for value in form[offset...] {
            guard let scalar = Unicode.Scalar(value) else { return }
            piece.append(scalar)
            if value != TdtVocabularyBias.marker { letters += 1 }
            let text = String(piece)
            // The bare `▁` piece narrows nothing; a standing bonus on it sprinkles word breaks.
            guard text != "\u{2581}", let ids = pieces.idsByPiece[text] else { continue }
            if offset == 0, form.first == TdtVocabularyBias.marker, letters < freshStartMinLetters { continue }
            for id in ids where entry.boost > best[id, default: -.infinity] {
                best[id] = entry.boost
            }
        }
    }
}

/// sherpa-onnx `ContextGraph` (v1.13.4, context-graph.cc) over fixed token ids, non-strict mode as
/// the NeMo modified beam search calls it.
struct ContextGraphIndex: Sendable {
    struct Node: Sendable {
        var token: Int
        var tokenScore: Float
        var nodeScore: Float
        var outputScore: Float
        var isEnd: Bool
        var next: [Int: Int] = [:]
        var fail: Int = 0
        var output: Int = -1
    }

    let nodes: [Node]
    let boost: Float

    init?(terms: [CustomVocabularyTerm], pieces: PieceTable, boost: Float, freshStartMinLetters: Int) {
        var nodes = [Node(token: -1, tokenScore: 0, nodeScore: 0, outputScore: 0, isEnd: false)]
        var built = 0
        for term in terms {
            guard let ids = term.tokenIds, !ids.isEmpty else { continue }
            built += 1
            var node = 0
            for (j, token) in ids.enumerated() {
                // Fresh-word guard: a root arc on a piece with fewer letters than the guard is
                // tracked but earns nothing, so it is never boosted open.
                let letters = pieces.scalarsById[token].map(PieceTable.letters) ?? 0
                let score: Float = (node == 0 && letters < freshStartMinLetters) ? 0 : boost
                let isLast = j == ids.count - 1
                if let child = nodes[node].next[token] {
                    let tokenScore = max(score, nodes[child].tokenScore)
                    nodes[child].tokenScore = tokenScore
                    let nodeScore = nodes[node].nodeScore + tokenScore
                    nodes[child].nodeScore = nodeScore
                    let isEnd = isLast || nodes[child].isEnd
                    nodes[child].outputScore = isEnd ? nodeScore : 0
                    nodes[child].isEnd = isEnd
                    node = child
                } else {
                    let nodeScore = nodes[node].nodeScore + score
                    nodes.append(
                        Node(
                            token: token, tokenScore: score, nodeScore: nodeScore,
                            outputScore: isLast ? nodeScore : 0, isEnd: isLast))
                    nodes[node].next[token] = nodes.count - 1
                    node = nodes.count - 1
                }
            }
        }
        guard built > 0 else { return nil }
        // FillFailOutput: breadth first.
        var queue: [Int] = []
        for (_, child) in nodes[0].next {
            nodes[child].fail = 0
            queue.append(child)
        }
        var head = 0
        while head < queue.count {
            let current = queue[head]
            head += 1
            for (token, child) in nodes[current].next {
                var fail = nodes[current].fail
                if let f = nodes[fail].next[token] {
                    fail = f
                } else {
                    fail = nodes[fail].fail
                    while nodes[fail].next[token] == nil {
                        fail = nodes[fail].fail
                        if nodes[fail].token == -1 { break }
                    }
                    if let f = nodes[fail].next[token] { fail = f }
                }
                nodes[child].fail = fail
                var output = fail
                while !nodes[output].isEnd {
                    output = nodes[output].fail
                    if nodes[output].token == -1 {
                        output = -1
                        break
                    }
                }
                nodes[child].output = output
                nodes[child].outputScore += output == -1 ? 0 : nodes[output].outputScore
                queue.append(child)
            }
        }
        self.nodes = nodes
        self.boost = boost
    }

    /// `ForwardOneStep(state, token, strict_mode: false)`.
    func forward(from state: Int, token: Int) -> (score: Float, next: Int, engaged: Bool) {
        var node: Int
        var score: Float
        if let child = nodes[state].next[token] {
            node = child
            score = nodes[child].tokenScore
        } else {
            node = nodes[state].fail
            while nodes[node].next[token] == nil {
                node = nodes[node].fail
                if nodes[node].token == -1 { break }
            }
            if let child = nodes[node].next[token] { node = child }
            score = nodes[node].nodeScore - nodes[state].nodeScore
        }
        if nodes[node].outputScore != 0 {
            let n = nodes[node]
            let outputScore = n.isEnd ? n.nodeScore : (n.output >= 0 ? nodes[n.output].nodeScore : n.nodeScore)
            return (score + outputScore - n.nodeScore, 0, true)
        }
        return (score + nodes[node].outputScore, node, node != 0)
    }

    /// What the beam search adds to a path that takes `token` from `state`: the in-place bonus on
    /// every arc out of the state (before top-K), plus the graph step. `engaged` is false when the
    /// token neither continues nor restarts a match (it only pays back), so it never outranks the
    /// plain argmax on the bias's account.
    func stepScore(from state: Int, token: Int) -> (score: Float, engaged: Bool) {
        let step = forward(from: state, token: token)
        var bonus: Float = 0
        if let child = nodes[state].next[token], nodes[child].tokenScore > 0 { bonus = boost }
        return (bonus + step.score, step.engaged)
    }
}
