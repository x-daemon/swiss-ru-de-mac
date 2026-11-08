// audiotap - writes the Mac's system audio (everything it plays) to standard output as
// 16 kHz mono float32 samples, for swiss-ru-de. Uses ScreenCaptureKit (macOS 13 or newer);
// the sound keeps playing normally.
//
// Build:  xcrun swiftc -O mac/audiotap.swift -o bin/audiotap
// Exit codes: 2 = capture stopped, 3 = no permission, 4 = no display, 5 = macOS too old.

import CoreMedia
import Foundation
import ScreenCaptureKit

let targetRate = 16_000.0

func fail(_ message: String, code: Int32) -> Never {
    FileHandle.standardError.write("audiotap: \(message)\n".data(using: .utf8)!)
    exit(code)
}

@available(macOS 13.0, *)
final class Tap: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let audioQueue = DispatchQueue(label: "audiotap.audio")
    private let videoQueue = DispatchQueue(label: "audiotap.video")
    private let out = FileHandle.standardOutput
    // linear resampler state, used only if the system does not deliver 16 kHz
    private var pending: [Float] = []
    private var position = 0.0

    func start() async throws {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        } catch {
            fail("permission: \(error.localizedDescription)", code: 3)
        }
        guard let display = content.displays.first else { fail("no display found", code: 4) }
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = Int(targetRate)
        config.channelCount = 1
        // ScreenCaptureKit always records the screen too: keep that tiny and rare.
        config.width = 160
        config.height = 90
        config.minimumFrameInterval = CMTime(value: 1, timescale: 2)
        config.queueDepth = 3
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: audioQueue)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: videoQueue)
        try await stream.startCapture()
        self.stream = stream
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferIsValid(sampleBuffer),
              let format = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee,
              asbd.mFormatID == kAudioFormatLinearPCM,
              asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              asbd.mBitsPerChannel == 32 else { return }
        let frames = CMSampleBufferGetNumSamples(sampleBuffer)
        let channels = max(1, Int(asbd.mChannelsPerFrame))
        guard frames > 0 else { return }

        var needed = 0
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, bufferListSizeNeededOut: &needed, bufferListOut: nil, bufferListSize: 0,
            blockBufferAllocator: nil, blockBufferMemoryAllocator: nil, flags: 0, blockBufferOut: nil)
        let raw = UnsafeMutableRawPointer.allocate(byteCount: needed,
                                                   alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        let list = raw.bindMemory(to: AudioBufferList.self, capacity: 1)
        var block: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, bufferListSizeNeededOut: nil, bufferListOut: list, bufferListSize: needed,
            blockBufferAllocator: nil, blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, blockBufferOut: &block)
        guard status == noErr else { return }

        // down-mix to mono
        var mono = [Float](repeating: 0, count: frames)
        let buffers = UnsafeMutableAudioBufferListPointer(list)
        if asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved == 0 {
            guard let data = buffers[0].mData else { return }
            let samples = data.assumingMemoryBound(to: Float.self)
            let n = min(frames, Int(buffers[0].mDataByteSize) / 4 / channels)
            for i in 0..<n {
                var sum: Float = 0
                for c in 0..<channels { sum += samples[i * channels + c] }
                mono[i] = sum / Float(channels)
            }
        } else {
            var used: Float = 0
            for buffer in buffers {
                guard let data = buffer.mData else { continue }
                let samples = data.assumingMemoryBound(to: Float.self)
                for i in 0..<min(frames, Int(buffer.mDataByteSize) / 4) { mono[i] += samples[i] }
                used += 1
            }
            if used > 1 { for i in 0..<frames { mono[i] /= used } }
        }
        write(resample(mono, from: asbd.mSampleRate))
    }

    private func resample(_ input: [Float], from rate: Double) -> [Float] {
        if abs(rate - targetRate) < 1 { return input }
        pending.append(contentsOf: input)
        let step = rate / targetRate
        var output: [Float] = []
        output.reserveCapacity(Int(Double(pending.count) / step) + 1)
        while position + 1 < Double(pending.count) {
            let i = Int(position)
            let f = Float(position - Double(i))
            output.append(pending[i] * (1 - f) + pending[i + 1] * f)
            position += step
        }
        let consumed = Int(position)
        pending.removeFirst(consumed)
        position -= Double(consumed)
        return output
    }

    private func write(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        do { try out.write(contentsOf: data) } catch { exit(0) }  // swiss-ru-de has stopped
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fail("capture stopped: \(error.localizedDescription)", code: 2)
    }
}

if CommandLine.arguments.contains("--version") {
    print("audiotap 1.0")
    exit(0)
}
guard #available(macOS 13.0, *) else { fail("needs macOS 13 (Ventura) or newer", code: 5) }
let tap = Tap()
Task {
    do {
        try await tap.start()
    } catch {
        let declined = (error as NSError).code == -3801  // SCStreamError.userDeclined
        fail((declined ? "permission: " : "could not start: ") + error.localizedDescription,
             code: declined ? 3 : 2)
    }
}
dispatchMain()
