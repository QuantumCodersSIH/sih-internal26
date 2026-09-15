from pathlib import Path 
import pandas as pd 
import random 
import shutil 
import subprocess 
import sys 
 
ROOT = Path.cwd() 
ARCHIVE = ROOT / "archive(2)" 
TRAIN_CSV = ROOT / "data" / "splits" / "train.csv" 
BACKUP_CSV = ROOT / "data" / "splits" / "train_before_expansion.csv" 
EXPANDED_CSV = ROOT / "data" / "splits" / "train_expanded.csv" 
 
random.seed(42) 
 
# Backup original split 
if not BACKUP_CSV.exists(): 
    shutil.copy2(TRAIN_CSV, BACKUP_CSV) 
 
df = pd.read_csv(TRAIN_CSV) 
 
print(f"Original training images: {len(df)}") 
print(f"Original columns: {list(df.columns)}") 
 
# Detect path/label columns 
path_col = "image" if "image" in df.columns else df.columns[0] 
label_col = "label" if "label" in df.columns else df.columns[1] 
 
new_rows = [] 
 
extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"} 
 
for dataset_num in range(1, 5): 
    base = ARCHIVE / f"Data Set {dataset_num}" / f"Data Set {dataset_num}" / "train" 
 
    real_dir = base / "real" 
    fake_dir = base / "fake" 
 
    real = [p for p in real_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions] 
    fake = [p for p in fake_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions] 
 
    random.shuffle(real) 
    random.shuffle(fake) 
 
    n = min(5000, len(real), len(fake)) 
 
    print(f"Data Set {dataset_num}: REAL={len(real)}, FAKE={len(fake)}, using {n}+{n}") 
 
    for p in real[:n]: 
        new_rows.append({ 
            path_col: p.relative_to(ROOT).as_posix(), 
            label_col: 0, 
        }) 
 
    for p in fake[:n]: 
        new_rows.append({ 
            path_col: p.relative_to(ROOT).as_posix(), 
            label_col: 1, 
        }) 
 
extra = pd.DataFrame(new_rows) 
 
expanded = pd.concat([df, extra], ignore_index=True) 
expanded.to_csv(EXPANDED_CSV, index=False) 
 
# Make trainer use expanded training data 
 
 
print() 
print("=" * 60) 
print(f"ADDED: {len(extra)} images") 
print(f"FINAL TRAINING SET: {len(expanded)} images") 
print(f"REAL added: {(extra[label_col] == 0).sum()}") 
print(f"FAKE added: {(extra[label_col] == 1).sum()}") 
print("=" * 60) 
print() 
print("Starting FINAL SignalScope training...") 
