#!/usr/bin/env python3
"""
Extract dictionary from AV-HuBERT checkpoint
"""
import torch
import tempfile
from pathlib import Path

MODEL_PATH = '/Users/yashvi/Desktop/HackSC/backend/models/avhubert/avhubert_base_lrs3_433h.pt'
WORK_DIR = Path(tempfile.gettempdir()) / "avhubert_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EXTRACTING DICTIONARY FROM CHECKPOINT")
print("=" * 70)

# Load checkpoint
print(f"Loading checkpoint: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

# Look for dictionary in various places
dict_found = False

# Check if there's a task dictionary
if 'cfg' in checkpoint:
    cfg = checkpoint['cfg']
    print(f"\nConfig type: {type(cfg)}")
    if hasattr(cfg, 'task'):
        print(f"Task config: {cfg.task}")

# Check for target_dictionary
if 'target_dictionary' in checkpoint:
    print("\n✓ Found 'target_dictionary' in checkpoint")
    target_dict = checkpoint['target_dictionary']
    
    # Save dictionary
    dict_path = WORK_DIR / "dict.ltr.txt"
    
    # If it's a fairseq Dictionary object
    if hasattr(target_dict, 'indices'):
        with open(dict_path, 'w') as f:
            for symbol, idx in sorted(target_dict.indices.items(), key=lambda x: x[1]):
                # Get count if available
                count = target_dict.count[idx] if hasattr(target_dict, 'count') else 1
                f.write(f"{symbol} {count}\n")
        print(f"✓ Saved dictionary to: {dict_path}")
        dict_found = True
    else:
        print(f"Dictionary type: {type(target_dict)}")
        print("Dictionary doesn't have expected format")

# Alternative: check for task state
if not dict_found and 'task_state' in checkpoint:
    print("\n✓ Found 'task_state' in checkpoint")
    task_state = checkpoint['task_state']
    print(f"Task state keys: {list(task_state.keys())}")
    
    # Try to get dictionaries from task_state
    if 'target_dictionary' in task_state:
        print("\n✓✓ Found 'target_dictionary' in task_state!")
        target_dict = task_state['target_dictionary']
        print(f"Dictionary type: {type(target_dict)}")
        
        dict_path = WORK_DIR / "dict.ltr.txt"
        
        # If it's a fairseq Dictionary object
        if hasattr(target_dict, 'indices'):
            with open(dict_path, 'w') as f:
                for symbol, idx in sorted(target_dict.indices.items(), key=lambda x: x[1]):
                    count = target_dict.count[idx] if hasattr(target_dict, 'count') else 1
                    f.write(f"{symbol} {count}\n")
            print(f"✓ Saved dictionary to: {dict_path}")
            print(f"✓ Dictionary size: {len(target_dict.indices)} symbols")
            dict_found = True
            
            # Show first 10 entries
            print("\nFirst 10 dictionary entries:")
            for i, (symbol, idx) in enumerate(sorted(target_dict.indices.items(), key=lambda x: x[1])[:10]):
                print(f"  {idx}: {repr(symbol)}")
        elif hasattr(target_dict, '__dict__'):
            print(f"Dictionary attributes: {list(vars(target_dict).keys())}")
        else:
            print("Dictionary doesn't have expected format")
    
    if not dict_found and 'dictionaries' in task_state:
        print("\n✓✓ Found 'dictionaries' in task_state!")
        dictionaries = task_state['dictionaries']
        print(f"Dictionaries type: {type(dictionaries)}")
        if isinstance(dictionaries, list) and len(dictionaries) > 0:
            target_dict = dictionaries[0]  # Use first dictionary
            print(f"First dictionary type: {type(target_dict)}")
            
            dict_path = WORK_DIR / "dict.ltr.txt"
            
            if hasattr(target_dict, 'indices'):
                with open(dict_path, 'w') as f:
                    for symbol, idx in sorted(target_dict.indices.items(), key=lambda x: x[1]):
                        count = target_dict.count[idx] if hasattr(target_dict, 'count') else 1
                        f.write(f"{symbol} {count}\n")
                print(f"✓ Saved dictionary to: {dict_path}")
                print(f"✓ Dictionary size: {len(target_dict.indices)} symbols")
                dict_found = True
                
                # Show first 10 entries
                print("\nFirst 10 dictionary entries:")
                for i, (symbol, idx) in enumerate(sorted(target_dict.indices.items(), key=lambda x: x[1])[:10]):
                    print(f"  {idx}: {repr(symbol)}")

# If still not found, look in args
if not dict_found and 'args' in checkpoint:
    print("\n✓ Found 'args' in checkpoint")
    args = checkpoint['args']
    print(f"Args type: {type(args)}")
    if hasattr(args, '__dict__'):
        print(f"Some args: {list(vars(args).keys())[:10]}")

if not dict_found:
    print("\n" + "=" * 70)
    print("❌ DICTIONARY NOT FOUND IN CHECKPOINT")
    print("=" * 70)
    print("\nCreating a default English character dictionary instead...")
    
    # Create default dictionary
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
    
    dict_path = WORK_DIR / "dict.ltr.txt"
    with open(dict_path, 'w') as f:
        f.write(dict_content.strip())
    
    print(f"✓ Created default dictionary at: {dict_path}")
    print("Note: This is a basic character dictionary. Decoding may not be perfect.")
    print("For better results, you need the actual dictionary used during training.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)