#!/usr/bin/env python3
"""
Fix dictionary format - the issue might be with how fairseq reads it
"""
import tempfile
from pathlib import Path

WORK_DIR = Path(tempfile.gettempdir()) / "avhubert_work"

print("=" * 70)
print("FIXING DICTIONARY FORMAT")
print("=" * 70)

# Read the wrd dictionary
wrd_dict = WORK_DIR / "dict.wrd.txt"

if not wrd_dict.exists():
    print(f"❌ {wrd_dict} not found")
    print("Run setup_files.py first!")
    exit(1)

with open(wrd_dict, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original file: {len(lines)} lines")
print("\nFirst 10 lines:")
for i, line in enumerate(lines[:10]):
    print(f"  {i}: {repr(line)}")

# The issue might be that Dictionary.add_from_file expects a specific format
# Let's check the fairseq source - it might be looking for a special marker

# Check if any line contains the special marker
has_marker = any('#fairseq:overwrite' in line for line in lines)
print(f"\nHas #fairseq:overwrite marker: {has_marker}")

# Check for actual duplicates in the file
words = {}
duplicates = []
for i, line in enumerate(lines):
    if line.strip():
        parts = line.split()
        if len(parts) >= 1:
            word = parts[0]
            if word in words:
                duplicates.append((i, word, words[word]))
            else:
                words[word] = i

if duplicates:
    print(f"\n❌ Found {len(duplicates)} duplicates:")
    for line_num, word, first_occurrence in duplicates[:10]:
        print(f"  Line {line_num}: '{word}' (first at line {first_occurrence})")
else:
    print("\n✓ No duplicates found in file")

# The real issue might be the recursive call in fairseq
# Let's check if there's a file path IN the dictionary file (which would cause recursion)
print("\n" + "=" * 70)
print("CHECKING FOR FILE PATHS IN DICTIONARY")
print("=" * 70)

for i, line in enumerate(lines):
    if '/' in line or '.txt' in line or 'dict.' in line:
        print(f"⚠️  Line {i} contains path-like string: {repr(line.rstrip())}")

print("\n" + "=" * 70)
print("ATTEMPTING TO USE FAIRSEQ DICTIONARY")
print("=" * 70)

# Try to load it with fairseq directly to see the actual error
try:
    from fairseq.data import Dictionary
    
    test_dict = Dictionary()
    print("Created empty Dictionary object")
    
    print(f"Attempting to load from: {wrd_dict}")
    test_dict = Dictionary.load(str(wrd_dict))
    
    print(f"✓ SUCCESS! Dictionary loaded with {len(test_dict)} entries")
    
except Exception as e:
    print(f"❌ Failed to load: {e}")
    import traceback
    traceback.print_exc()
    
    # If it failed, try to understand why by reading the fairseq source
    print("\n" + "=" * 70)
    print("CHECKING FAIRSEQ DICTIONARY SOURCE")
    print("=" * 70)
    
    import inspect
    print(f"Dictionary.load location: {inspect.getfile(Dictionary.load)}")
    print(f"Dictionary.add_from_file location: {inspect.getfile(Dictionary.add_from_file)}")