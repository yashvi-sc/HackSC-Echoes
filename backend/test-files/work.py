#!/usr/bin/env python3
"""
Diagnostic script to check AV-HuBERT installation
"""
import os
import sys
from pathlib import Path

# UPDATE THESE PATHS
AVHUBERT_ROOT = '/Users/yashvi/Desktop/HackSC/backend/av_hubert'

print("=" * 70)
print("AV-HUBERT DIAGNOSTIC SCRIPT")
print("=" * 70)

# Check 1: Directory structure
print("\n[CHECK 1] Directory Structure")
print(f"AV-HuBERT root: {AVHUBERT_ROOT}")
print(f"Exists: {os.path.exists(AVHUBERT_ROOT)}")

if os.path.exists(AVHUBERT_ROOT):
    print("\nContents of AV-HuBERT root:")
    for item in os.listdir(AVHUBERT_ROOT):
        print(f"  - {item}")
    
    # Check for avhubert subdirectory
    avhubert_subdir = os.path.join(AVHUBERT_ROOT, 'avhubert')
    if os.path.exists(avhubert_subdir):
        print(f"\nContents of {avhubert_subdir}:")
        for item in os.listdir(avhubert_subdir):
            print(f"  - {item}")
        
        # Check for key files
        key_files = ['hubert.py', 'hubert_pretraining.py', '__init__.py']
        print("\nKey files:")
        for f in key_files:
            fpath = os.path.join(avhubert_subdir, f)
            exists = os.path.exists(fpath)
            print(f"  {f}: {'✓' if exists else '✗'}")

# Check 2: Import test
print("\n[CHECK 2] Import Test")
sys.path.insert(0, AVHUBERT_ROOT)

try:
    print("Importing fairseq...")
    from fairseq import models as fairseq_models
    print("✓ Fairseq imported")
    print(f"Models before AV-HuBERT: {[k for k in fairseq_models.MODEL_REGISTRY.keys() if 'hubert' in k.lower()]}")
except Exception as e:
    print(f"✗ Failed to import fairseq: {e}")
    sys.exit(1)

try:
    print("\nImporting hubert_pretraining...")
    sys.path.insert(0, os.path.join(AVHUBERT_ROOT, 'avhubert'))
    import hubert_pretraining
    print("✓ hubert_pretraining imported")
except Exception as e:
    print(f"✗ Failed to import hubert_pretraining: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nImporting hubert module...")
    import hubert
    print("✓ hubert module imported")
    print(f"Models after import: {[k for k in fairseq_models.MODEL_REGISTRY.keys() if 'hubert' in k.lower()]}")
    
    # Check if av_hubert is registered
    if 'av_hubert' in fairseq_models.MODEL_REGISTRY:
        print("✓✓✓ SUCCESS: av_hubert is registered!")
        print(f"Model class: {fairseq_models.MODEL_REGISTRY['av_hubert']}")
    else:
        print("✗ av_hubert NOT in registry")
        print("This is the problem - the model decorator isn't running")
        
except Exception as e:
    print(f"✗ Failed to import hubert: {e}")
    import traceback
    traceback.print_exc()

# Check 3: Look for @register_model decorator
print("\n[CHECK 3] Checking for @register_model decorator")
hubert_py = os.path.join(AVHUBERT_ROOT, 'avhubert', 'hubert.py')
if os.path.exists(hubert_py):
    with open(hubert_py, 'r') as f:
        content = f.read()
        if '@register_model(' in content:
            print("✓ Found @register_model decorator in hubert.py")
            # Find the decorator line
            for i, line in enumerate(content.split('\n')):
                if '@register_model(' in line:
                    print(f"  Line {i+1}: {line.strip()}")
                    # Print next few lines to see class definition
                    lines = content.split('\n')
                    for j in range(i, min(i+5, len(lines))):
                        print(f"  Line {j+1}: {lines[j]}")
                    break
        else:
            print("✗ No @register_model decorator found")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)