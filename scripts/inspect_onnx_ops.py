"""
Inspect ONNX Operators for Raspberry Pi (ARM64) Compatibility
Checks all op types in the exported GTCRN ONNX model against standard ARM64 CPUExecutionProvider capabilities.
"""
import onnx
from collections import Counter

def check_onnx_operators(onnx_path="checkpoints/gtcrn_simple.onnx"):
    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    op_counts = Counter()

    for node in model.graph.node:
        op_counts[node.op_type] += 1

    print("\n--- Operator Breakdown in GTCRN ONNX Model ---")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op:20s}: {count}")

    # Standard ARM64 CPUExecutionProvider compatibility check
    # Reference: ONNX Runtime CPU EP supports all default ONNX opset operators
    supported_ops = {
        "Conv", "ConvTranspose", "MatMul", "Add", "Sub", "Mul", "Div", "Sqrt",
        "Tanh", "Sigmoid", "PRelu", "Relu", "Reshape", "Transpose", "Concat",
        "Split", "Slice", "Gather", "Shape", "Unsqueeze", "Squeeze", "Pad",
        "ReduceMean", "Pow", "Gemm", "Identity"
    }

    all_supported = True
    print("\n--- ARM64 Compatibility Check ---")
    for op in op_counts:
        if op in supported_ops:
            print(f"  [COMPATIBLE] {op:15s} (Fully supported on ARM64 Cortex-A72 / A76)")
        else:
            print(f"  [VERIFY]     {op:15s} (Check specific opset version)")
            all_supported = False

    if all_supported:
        print("\nAll model operators are standard, native ONNX Runtime CPU operators supported on Raspberry Pi OS 64-bit (ARM64)!")
    return op_counts

if __name__ == "__main__":
    check_onnx_operators()
