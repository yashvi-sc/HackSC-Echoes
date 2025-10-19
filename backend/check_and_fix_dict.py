#!/usr/bin/env python3
"""
Check and fix the dictionary file
"""
import tempfile
from pathlib import Path
from collections import Counter

WORK_DIR = Path(tempfile.gettempdir()) / "avhubert_work"
dict_path = WORK_DIR / "dict.ltr.txt"

print("=" * 70)
print("CHECKING DICTIONARY FILE")
print("=" * 70)

if not dict_path.exists():
    print(f"❌ Dictionary file not found: {dict_path}")
    exit(1)

print(f"✓ Found dictionary at: {dict_path}")

# Read and analyze the file
with open(dict_path, 'r') as f:
    lines = f.readlines()

print(f"\nTotal lines: {len(lines)}")
print(f"\nFirst 10 lines:")
for i, line in enumerate(lines[:10]):
    print(f"  {i}: {line.rstrip()}")

# Check for duplicates
words = [line.split()[0] if line.strip() else '' for line in lines]
word_counts = Counter(words)
duplicates = {word: count for word, count in word_counts.items() if count > 1}

if duplicates:
    print(f"\n❌ FOUND DUPLICATES:")
    for word, count in duplicates.items():
        print(f"  '{word}' appears {count} times")
    
    print("\n🔧 FIXING: Removing duplicates (keeping first occurrence)...")
    
    seen = set()
    clean_lines = []
    for line in lines:
        if line.strip():
            word = line.split()[0]
            if word not in seen:
                clean_lines.append(line)
                seen.add(word)
    
    # Backup original
    backup_path = dict_path.with_suffix('.txt.backup')
    print(f"\n📦 Backing up original to: {backup_path}")
    with open(backup_path, 'w') as f:
        f.writelines(lines)
    
    # Write cleaned version
    print(f"✍️  Writing cleaned dictionary...")
    with open(dict_path, 'w') as f:
        f.writelines(clean_lines)
    
    print(f"\n✓ Fixed! Reduced from {len(lines)} to {len(clean_lines)} lines")
    print(f"\nFirst 10 lines of cleaned file:")
    for i, line in enumerate(clean_lines[:10]):
        print(f"  {i}: {line.rstrip()}")
else:
    print("\n✓ No duplicates found! Dictionary is clean.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)