#!/usr/bin/env python3
"""
Check what's in fairseq's Dictionary.add_from_file method
"""

fairseq_dict_file = "/Users/yashvi/Desktop/HackSC/backend/venvten/lib/python3.10/site-packages/fairseq/data/dictionary.py"

print("=" * 70)
print("FAIRSEQ DICTIONARY SOURCE CODE")
print("=" * 70)

with open(fairseq_dict_file, 'r') as f:
    lines = f.readlines()

# Find add_from_file method
in_method = False
method_lines = []
indent_count = 0

for i, line in enumerate(lines, 1):
    if 'def add_from_file(' in line:
        in_method = True
        indent_count = len(line) - len(line.lstrip())
        print(f"\nFound add_from_file at line {i}")
        print("=" * 70)
    
    if in_method:
        # Check if we've exited the method (dedented to same or less)
        if line.strip() and not line.strip().startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_count and len(method_lines) > 1:
                break
        
        method_lines.append((i, line))

# Print the method
for line_num, line in method_lines[:50]:  # First 50 lines of method
    print(f"{line_num:4d}: {line.rstrip()}")

print("\n" + "=" * 70)

# Look for line 237 specifically
if len(lines) >= 237:
    print(f"\nLine 237:")
    print(f"  {lines[236].rstrip()}")
    print("\nContext around line 237:")
    for i in range(max(0, 234), min(len(lines), 242)):
        marker = ">>>" if i == 236 else "   "
        print(f"{marker} {i+1:4d}: {lines[i].rstrip()}")