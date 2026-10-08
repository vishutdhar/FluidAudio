// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5. It
// holds scripted decoder and joint models for the decode-time vocabulary bias tests.

@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// One scripted joint step: the plain argmax, its probability, the duration bin and the top-K the
/// joint returns beside it.
struct ScriptedJointStep {
    let token: Int
    let probability: Float
    let durationBin: Int
    let topK: [(id: Int, logit: Float)]
}

/// The keys a decode read from the joint's outputs, shared by every provider the joint returns.
final class ScriptedReadLog {
    private(set) var keys: Set<String> = []
    func record(_ key: String) { keys.insert(key) }
}

/// A joint output provider that records which outputs the decoder reads, so a test can tell a
/// decode that extracted top-K from one that did not.
final class ScriptedJointOutput: NSObject, MLFeatureProvider {
    private let values: [String: MLFeatureValue]
    private let log: ScriptedReadLog

    init(values: [String: MLFeatureValue], log: ScriptedReadLog) {
        self.values = values
        self.log = log
        super.init()
    }

    var featureNames: Set<String> { Set(values.keys) }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        log.record(featureName)
        return values[featureName]
    }
}

/// The decoder LSTM, scripted: its projection carries the token it was fed in element 0, so the
/// scripted joint can see the decoder's context. Like the real model it writes its new state into
/// the output backings the decoder hands it, which are the state's own arrays.
final class ScriptedDecoderModel: MLModel {
    private(set) var fedTokens: [Int] = []

    override func prediction(
        from input: MLFeatureProvider, options: MLPredictionOptions = MLPredictionOptions()
    ) throws -> MLFeatureProvider {
        guard let targets = input.featureValue(for: "targets")?.multiArrayValue else {
            throw ASRError.processingFailed("scripted decoder: no targets")
        }
        let token = targets[0].intValue
        fedTokens.append(token)
        let projection = try MLMultiArray(
            shape: [1, NSNumber(value: ASRConstants.decoderHiddenSize), 1], dataType: .float32)
        projection.resetData(to: 0)
        projection[0] = NSNumber(value: Float(token))
        var outputs: [String: MLFeatureValue] = ["decoder": MLFeatureValue(multiArray: projection)]
        for name in ["h_out", "c_out"] {
            if let backing = options.outputBackings[name] as? MLMultiArray {
                backing[0] = NSNumber(value: Float(token))
                outputs[name] = MLFeatureValue(multiArray: backing)
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: outputs)
    }
}

/// The joint, scripted by encoder frame and the token the decoder was last fed. Every call's
/// frame is recorded, so a test can see where the decoder stood.
final class ScriptedJointModel: MLModel {
    private let script: (_ frame: Int, _ lastToken: Int) -> ScriptedJointStep
    let readLog = ScriptedReadLog()
    private(set) var frames: [Int] = []

    init(script: @escaping (_ frame: Int, _ lastToken: Int) -> ScriptedJointStep) {
        self.script = script
        super.init()
    }

    override func prediction(
        from input: MLFeatureProvider, options: MLPredictionOptions = MLPredictionOptions()
    ) throws -> MLFeatureProvider {
        guard let encoderStep = input.featureValue(for: "encoder_step")?.multiArrayValue,
            let decoderStep = input.featureValue(for: "decoder_step")?.multiArrayValue
        else {
            throw ASRError.processingFailed("scripted joint: missing inputs")
        }
        let frame = Int(encoderStep[0].floatValue.rounded())
        let lastToken = Int(decoderStep[0].floatValue.rounded())
        frames.append(frame)
        let step = script(frame, lastToken)
        func scalar(_ value: Int32) throws -> MLMultiArray {
            let array = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
            array[0] = NSNumber(value: value)
            return array
        }
        let probability = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        probability[0] = NSNumber(value: step.probability)
        let ids = try MLMultiArray(shape: [NSNumber(value: step.topK.count)], dataType: .int32)
        let logits = try MLMultiArray(shape: [NSNumber(value: step.topK.count)], dataType: .float32)
        for (i, entry) in step.topK.enumerated() {
            ids[i] = NSNumber(value: Int32(entry.id))
            logits[i] = NSNumber(value: entry.logit)
        }
        return ScriptedJointOutput(
            values: [
                "token_id": MLFeatureValue(multiArray: try scalar(Int32(step.token))),
                "token_prob": MLFeatureValue(multiArray: probability),
                "duration": MLFeatureValue(multiArray: try scalar(Int32(step.durationBin))),
                "top_k_ids": MLFeatureValue(multiArray: ids),
                "top_k_logits": MLFeatureValue(multiArray: logits),
            ], log: readLog)
    }
}

enum ScriptedTdt {
    static let blank = 8192

    /// A config whose encoder frames are one value wide: frame `t` holds `t`, which the scripted
    /// joint reads back.
    static let config = ASRConfig(encoderHiddenSize: 1)

    static func encoderOutput(frames: Int) throws -> MLMultiArray {
        let output = try MLMultiArray(shape: [1, NSNumber(value: frames), 1], dataType: .float32)
        for t in 0..<frames { output[t] = NSNumber(value: Float(t)) }
        return output
    }

    /// The top-K softmax of `logit`, the probability the decoder gives a flipped token.
    static func topKSoftmax(_ logit: Float, in logits: [Float]) -> Float {
        let maxLogit = logits.max() ?? 0
        let sum = logits.reduce(Float(0)) { $0 + expf($1 - maxLogit) }
        return expf(logit - maxLogit) / sum
    }
}
