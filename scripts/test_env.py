import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

packages = [
    "torch",
    "torchaudio",
    "soundfile",
    "onnxruntime",
    "einops",
    "ptflops",
    "pystoi",
    "pesq",
    "numpy",
    "scipy"
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  [OK] {pkg} (version: {ver})")
    except ImportError as e:
        print(f"  [MISSING] {pkg}: {e}")
    except Exception as e:
        print(f"  [ERROR] {pkg}: {e}")
