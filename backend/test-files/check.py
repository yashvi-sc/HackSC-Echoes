#!/usr/bin/env python3
"""
Check the build_encoder_layer method in fairseq
"""

fairseq_file = "/Users/yashvi/Desktop/HackSC/backend/venvten/lib/python3.10/site-packages/fairseq/models/wav2vec/wav2vec2.py"

print("=" * 70)
print("FAIRSEQ WAV2VEC2 SOURCE CODE")
print("=" * 70)

with open(fairseq_file, 'r') as f:
    lines = f.readlines()

# Find build_encoder_layer method
in_method = False
method_lines = []
indent_count = 0

for i, line in enumerate(lines, 1):
    if 'def build_encoder_layer(' in line:
        in_method = True
        indent_count = len(line) - len(line.lstrip())
        print(f"\nFound build_encoder_layer at line {i}")
        print("=" * 70)
    
    if in_method:
        # Check if we've exited the method
        if line.strip() and not line.strip().startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_count and len(method_lines) > 1:
                break
        
        method_lines.append((i, line))

# Print the method
for line_num, line in method_lines[:100]:  # First 100 lines
    marker = ">>>" if line_num == 944 else "   "
    print(f"{marker} {line_num:4d}: {line.rstrip()}")