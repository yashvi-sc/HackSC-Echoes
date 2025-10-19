#!/usr/bin/env python3
"""
Setup all required files for AV-HuBERT
"""
import shutil
import tempfile
from pathlib import Path

WORK_DIR = Path(tempfile.gettempdir()) / "avhubert_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SETTING UP REQUIRED FILES")
print("=" * 70)

# 1. Copy dict.ltr.txt to dict.wrd.txt (checkpoint uses 'wrd' labels)
ltr_dict = WORK_DIR / "dict.ltr.txt"
wrd_dict = WORK_DIR / "dict.wrd.txt"

if ltr_dict.exists():
    shutil.copy(ltr_dict, wrd_dict)
    print(f"✓ Copied {ltr_dict}")
    print(f"  -> {wrd_dict}")
else:
    print(f"❌ Source dictionary not found: {ltr_dict}")
    print("Run extract_dict_from_checkpoint.py first!")

# 2. Create a dummy sentencepiece model file (checkpoint expects this)
# The actual tokenization will use the dictionary, so this is just a placeholder
spm_dir = WORK_DIR / "spm"
spm_dir.mkdir(exist_ok=True)
spm_model = spm_dir / "spm_unigram1000.model"

# Create empty placeholder
spm_model.touch()
print(f"\n✓ Created placeholder: {spm_model}")
print("  (Not actually used for inference)")

print("\n" + "=" * 70)
print("SETUP COMPLETE")
print("=" * 70)
print("\nFiles created:")
print(f"  - {wrd_dict}")
print(f"  - {spm_model}")