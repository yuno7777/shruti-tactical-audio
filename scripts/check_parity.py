import soundfile as sf
import numpy as np

def check_outputs():
    pytorch_enh, fs1 = sf.read("samples/baseline_enhanced_pytorch.wav")
    onnx_enh, fs2 = sf.read("samples/baseline_enhanced_onnx_streaming.wav")
    min_len = min(len(pytorch_enh), len(onnx_enh))
    diff = np.abs(pytorch_enh[:min_len] - onnx_enh[:min_len])
    print(f"Max abs diff between PyTorch offline and ONNX streaming: {np.max(diff):.4f}")
    print(f"Mean abs diff: {np.mean(diff):.6f}")
    corr = np.corrcoef(pytorch_enh[:min_len], onnx_enh[:min_len])[0, 1]
    print(f"Correlation: {corr:.6f}")

if __name__ == "__main__":
    check_outputs()
