#!/usr/bin/env python3
"""
Create a minimal dictionary file for AV-HuBERT inference
"""
import os
import tempfile
from pathlib import Path

# Create work directory
WORK_DIR = Path(tempfile.gettempdir()) / "avhubert_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Create a simple English letter dictionary
# This is a character-level dictionary with common English letters
dict_content = """<s> 1
<pad> 1
</s> 1
<unk> 1
| 1
' 1
a 1000
b 800
c 700
d 650
e 1200
f 500
g 450
h 600
i 850
j 100
k 300
l 700
m 600
n 750
o 900
p 500
q 50
r 700
s 800
t 900
u 600
v 300
w 400
x 100
y 400
z 80
"""

# Write dictionary file
dict_path = WORK_DIR / "dict.ltr.txt"
with open(dict_path, 'w') as f:
    f.write(dict_content.strip())

print(f"✓ Created dictionary at: {dict_path}")
print(f"✓ Dictionary has {len(dict_content.strip().split(chr(10)))} entries")

# Verify it was created
if dict_path.exists():
    print("✓ File exists and is readable")
    with open(dict_path, 'r') as f:
        lines = f.readlines()
        print(f"✓ First 5 lines:")
        for line in lines[:5]:
            print(f"  {line.strip()}")
else:
    print("✗ Failed to create file")