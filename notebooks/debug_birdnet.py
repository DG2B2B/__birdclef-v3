# === CELL 1: BirdNET Diagnostic Notebook ===
# Objectif: trouver pourquoi BirdNET n'est pas accessible
# et tester l'inférence si le modèle est trouvé.

import os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

print("=" * 60)
print("BIRDNET DIAGNOSTIC NOTEBOOK")
print("=" * 60)

# ═══════════════════════════════════
# 1. Explore ALL /kaggle/input paths
# ═══════════════════════════════════
print("\n[1] Listing /kaggle/input structure...")

INPUT = Path("/kaggle/input")
for depth in [1, 2, 3, 4]:
    print(f"\n--- Depth {depth} ---")
    for d in sorted(INPUT.glob("*/" * depth)):
        rel = str(d.relative_to(INPUT))
        # Only show dirs that might be relevant
        if any(kw in rel.lower() for kw in ["bird", "model", "tflite", "analyzer", "shadi"]):
            print(f"  {rel}/")
            # List contents
            try:
                for f in sorted(d.iterdir()):
                    print(f"    {f.name}")
            except PermissionError:
                print("    [permission denied]")

# ═══════════════════════════════════
# 2. Search ALL .tflite files
# ═══════════════════════════════════
print("\n\n[2] Searching for ALL .tflite files in /kaggle/input...")
tflite_files = sorted(INPUT.rglob("*.tflite"))
print(f"  Found {len(tflite_files)} .tflite file(s):")
for f in tflite_files:
    print(f"    {f}")

# Also search for any file with 'birdnet' in name
print("\n[3] Searching for files with 'birdnet' in name...")
birdnet_files = sorted(INPUT.rglob("*birdnet*")) + sorted(INPUT.rglob("*BirdNET*"))
print(f"  Found {len(birdnet_files)} file(s):")
for f in birdnet_files:
    print(f"    {f}")

# ═══════════════════════════════════
# 3. Check models directory specifically
# ═══════════════════════════════════
print("\n[4] Checking /kaggle/input/models/ specifically...")
MODELS_DIR = Path("/kaggle/input/models")
if MODELS_DIR.exists():
    for d in sorted(MODELS_DIR.iterdir()):
        print(f"  {d.name}/")
        try:
            for sub in sorted(d.rglob("*"))[:20]:  # limit depth
                if sub.is_file():
                    print(f"    {sub.relative_to(MODELS_DIR)}")
        except:
            print(f"    [error listing]")
else:
    print("  /kaggle/input/models/ DOES NOT EXIST!")
    # Check what's in /kaggle/input/
    print("\n  Contents of /kaggle/input/:")
    for d in sorted(INPUT.iterdir()):
        print(f"    {d.name}/")

# ═══════════════════════════════════
# 4. Check environment variables
# ═══════════════════════════════════
print("\n[5] Kaggle environment...")
for var in ["KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DATA_PROXY_TOKEN", "KAGGLE_URL_BASE"]:
    print(f"  {var}={os.environ.get(var, 'NOT SET')}")

# ═══════════════════════════════════
# 5. Try loading BirdNET if found
# ═══════════════════════════════════
print("\n[6] Attempting BirdNET load...")

bn_path = None
if tflite_files:
    bn_path = tflite_files[0]
elif birdnet_files:
    bn_path = next((f for f in birdnet_files if f.suffix == '.tflite'), birdnet_files[0])

if bn_path:
    print(f"  Trying to load: {bn_path}")
    try:
        # Try tflite_runtime first
        from tflite_runtime.interpreter import Interpreter as TFLiteInterp
        print("  Using tflite_runtime")
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterp
            print("  Using tensorflow.lite")
        except ImportError:
            print("  ERROR: Neither tflite_runtime nor tensorflow available!")
            TFLiteInterp = None
    
    if TFLiteInterp:
        try:
            interp = TFLiteInterp(model_path=str(bn_path), num_threads=4)
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            out = interp.get_output_details()
            print(f"  SUCCESS! Model loaded.")
            print(f"  Input: {inp['shape']} ({inp['dtype']})")
            print(f"  Outputs: {len(out)}")
            for o in out:
                print(f"    {o['name']}: {o['shape']} ({o['dtype']})")
            
            # Test inference on random audio
            print("\n  Testing inference on random 3s audio...")
            test_audio = np.random.randn(1, 144000).astype(np.float32)
            interp.set_tensor(inp['index'], test_audio)
            interp.invoke()
            logits = interp.get_tensor(out[-1]['index'])[0]
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            print("  BirdNET IS WORKING!")
        except Exception as e:
            print(f"  ERROR loading model: {e}")
else:
    print("  No .tflite or BirdNET file found anywhere in /kaggle/input/")
    print("  CHECK: Is the model source correctly attached?")
    print("  Expected: shadiakiki1/birdnet-analyzer/tflite/birdnet_global_6k_v2.4_model_fp32-1/2")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
