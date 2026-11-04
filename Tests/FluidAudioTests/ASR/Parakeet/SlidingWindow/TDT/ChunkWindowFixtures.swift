// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias-windows of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5: the
// real recordings the window plan and decode ahead tests run on.

import AVFoundation
import Foundation
import XCTest

/// The three speech fixtures of the sliding window tests (16 kHz mono, 16.9 to 21.4 s), and recordings
/// made by joining them in different orders: two or three in a row give 36.7 to 58.1 s, three to five
/// windows of the Ultra layout, with their pauses falling at different places in each order. No sample is
/// generated: every recording is the fixtures' own audio.
enum ChunkWindowFixtures {
    static let names = [
        "01-validation-request-21.4s", "02-release-readiness-19.8s", "03-diff-explanation-16.9s",
    ]

    static func samples(of name: String) throws -> [Float] {
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

    /// The fixtures named, one after another.
    static func recording(_ order: [String]) throws -> [Float] {
        try order.map(samples(of:)).flatMap { $0 }
    }

    /// The three fixtures in a row, in the order given (the default is the numbered one).
    static func allThree(order: [Int] = [0, 1, 2]) throws -> [Float] {
        try recording(order.map { names[$0] })
    }

    /// Every order of two and of three fixtures, named by the fixtures' numbers.
    static func joinedRecordings() throws -> [(name: String, audio: [Float])] {
        let orders: [[Int]] = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
            [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],
        ]
        return try orders.map { order in
            (name: order.map { String($0 + 1) }.joined(separator: " then "), audio: try allThree(order: order))
        }
    }
}
