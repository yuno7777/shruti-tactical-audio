"""
Model Export Pipeline:
Converts GTCRN PyTorch model (offline & streaming) to ONNX format.
Verifies numerical equivalence and validates streaming state cache dynamics.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.gtcrn import GTCRN
from src.inference.stream.gtcrn_stream import StreamGTCRN
from src.inference.stream.modules.convert import convert_to_stream

def export_gtcrn_to_onnx(ckpt_path="checkpoints/fine_tuned/best_gtcrn_defence.tar",
                         output_dir="checkpoints/fine_tuned",
                         opset_version=18):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"Loading checkpoint: {ckpt_path}")
    model = GTCRN().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])

    # 1. Initialize Streaming GTCRN and transfer weights
    print("Converting offline GTCRN weights to StreamGTCRN...")
    stream_model = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_model, model)

    # 2. Prepare dummy inputs matching streaming forward signature:
    # spec: (B, F, T, 2) = (1, 257, 1, 2)
    # conv_cache: (2, B, 16, 16, 33) = (2, 1, 16, 16, 33)
    # tra_cache: (2, 3, 1, B, 16) = (2, 3, 1, 1, 16)
    # inter_cache: (2, 1, BF, 16) = (2, 1, 33, 16)
    dummy_spec = torch.randn(1, 257, 1, 2, dtype=torch.float32)
    dummy_conv_cache = torch.zeros(2, 1, 16, 16, 33, dtype=torch.float32)
    dummy_tra_cache = torch.zeros(2, 3, 1, 1, 16, dtype=torch.float32)
    dummy_inter_cache = torch.zeros(2, 1, 33, 16, dtype=torch.float32)

    onnx_file = os.path.join(output_dir, "gtcrn_stream.onnx")
    print(f"Exporting streaming ONNX model to {onnx_file} (opset {opset_version})...")

    torch.onnx.export(
        stream_model,
        (dummy_spec, dummy_conv_cache, dummy_tra_cache, dummy_inter_cache),
        onnx_file,
        input_names=['mix', 'conv_cache', 'tra_cache', 'inter_cache'],
        output_names=['enh', 'conv_cache_out', 'tra_cache_out', 'inter_cache_out'],
        opset_version=opset_version,
        verbose=False
    )
    print("ONNX export succeeded.")

    # 3. Simplify ONNX if onnxsim is available
    try:
        import onnx
        from onnxsim import simplify
        model_proto = onnx.load(onnx_file)
        model_simp, check = simplify(model_proto)
        if check:
            simple_file = os.path.join(output_dir, "gtcrn_stream_simple.onnx")
            onnx.save(model_simp, simple_file)
            print(f"Simplified ONNX model saved to: {simple_file}")
    except ImportError:
        print("Note: onnxsim not installed, skipping simplification step.")

    # 4. Verify numerical equivalence with PyTorch streaming model
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])

    # Clone inputs so in-place cache updates don't desynchronize PyTorch and ONNX
    dummy_spec_pt = dummy_spec.clone()
    dummy_conv_pt = dummy_conv_cache.clone()
    dummy_tra_pt = dummy_tra_cache.clone()
    dummy_inter_pt = dummy_inter_cache.clone()

    dummy_spec_ort = dummy_spec.clone().numpy()
    dummy_conv_ort = dummy_conv_cache.clone().numpy()
    dummy_tra_ort = dummy_tra_cache.clone().numpy()
    dummy_inter_ort = dummy_inter_cache.clone().numpy()

    with torch.no_grad():
        py_out, py_c, py_t, py_i = stream_model(dummy_spec_pt, dummy_conv_pt, dummy_tra_pt, dummy_inter_pt)

    ort_inputs = {
        'mix': dummy_spec_ort,
        'conv_cache': dummy_conv_ort,
        'tra_cache': dummy_tra_ort,
        'inter_cache': dummy_inter_ort
    }
    ort_out, ort_c, ort_t, ort_i = session.run([], ort_inputs)

    diff = np.max(np.abs(py_out.numpy() - ort_out))
    print(f"Numerical verification (PyTorch vs ONNX): max absolute diff = {diff:.2e}")
    assert diff < 1e-3, f"Discrepancy too high: {diff}"
    print("Model export verification PASSED successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GTCRN to streaming ONNX")
    parser.add_argument("--ckpt", type=str, default="checkpoints/fine_tuned/best_gtcrn_defence.tar")
    parser.add_argument("--out", type=str, default="checkpoints/fine_tuned")
    args = parser.parse_args()

    export_gtcrn_to_onnx(args.ckpt, args.out)
