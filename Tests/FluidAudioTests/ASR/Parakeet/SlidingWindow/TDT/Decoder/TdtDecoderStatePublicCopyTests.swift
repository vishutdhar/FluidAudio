// Koegaki change notice (Apache License 2.0, section 4(b)): this file was added for Koegaki on
// branch koegaki-bias of github.com/vishutdhar/FluidAudio, based on upstream tag v0.17.5. It
// imports FluidAudio without @testable, so it compiles only while `TdtDecoderState(from:)` is
// public: a caller outside the package copies the state before a decode it may repeat.

import FluidAudio
import XCTest

final class TdtDecoderStatePublicCopyTests: XCTestCase {
    func testTheCopyInitializerIsPublic() throws {
        let original = try TdtDecoderState()
        let copy = try TdtDecoderState(from: original)
        XCTAssertNotNil(copy)
    }
}
