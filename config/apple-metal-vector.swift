import Foundation
import Metal

let args = CommandLine.arguments
let n = args.count > 1 ? (Int(args[1]) ?? 16_777_216) : 16_777_216
let device = MTLCreateSystemDefaultDevice()
guard let gpu = device else {
    fputs("Metal device not available\n", stderr)
    exit(127)
}

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void vadd(const device float *a [[buffer(0)]],
                 const device float *b [[buffer(1)]],
                 device float *c [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
  c[id] = a[id] + b[id];
}
"""

let library = try gpu.makeLibrary(source: source, options: nil)
let function = library.makeFunction(name: "vadd")!
let pipeline = try gpu.makeComputePipelineState(function: function)
let queue = gpu.makeCommandQueue()!
let bytes = n * MemoryLayout<Float>.stride
let a = gpu.makeBuffer(length: bytes, options: .storageModeShared)!
let b = gpu.makeBuffer(length: bytes, options: .storageModeShared)!
let c = gpu.makeBuffer(length: bytes, options: .storageModeShared)!

let ap = a.contents().bindMemory(to: Float.self, capacity: n)
let bp = b.contents().bindMemory(to: Float.self, capacity: n)
let cp = c.contents().bindMemory(to: Float.self, capacity: n)
for i in 0..<n {
    ap[i] = Float(i & 255) * 0.5
    bp[i] = Float(i & 127) * 0.25
}

let command = queue.makeCommandBuffer()!
let encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(a, offset: 0, index: 0)
encoder.setBuffer(b, offset: 0, index: 1)
encoder.setBuffer(c, offset: 0, index: 2)
let w = pipeline.threadExecutionWidth
let threadsPerGroup = MTLSize(width: w, height: 1, depth: 1)
let threads = MTLSize(width: n, height: 1, depth: 1)
let started = DispatchTime.now().uptimeNanoseconds
encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerGroup)
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
let ended = DispatchTime.now().uptimeNanoseconds

var maxError: Float = 0.0
let samples = min(n, 1024)
for i in 0..<samples {
    let expected = ap[i] + bp[i]
    maxError = max(maxError, abs(expected - cp[i]))
}

let seconds = Double(ended - started) / 1.0e9
let gbps = Double(bytes * 3) / seconds / 1.0e9
print("Device: \(gpu.name)")
print("Elements: \(n)")
print(String(format: "Kernel time: %.6f s", seconds))
print(String(format: "Device approx bandwidth: %.6f GB/s", gbps))
print(String(format: "Max error: %.6f", Double(maxError)))
