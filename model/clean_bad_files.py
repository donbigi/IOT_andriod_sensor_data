import os
import glob
import pandas as pd

DATA_DIR = "data"   # folder with your CSV files
MIN_ROWS = 350
MAX_ROWS = 600

DRY_RUN = False  # ← change to False to actually delete files

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

to_delete = []

for path in files:
    try:
        df = pd.read_csv(path)
        L = len(df)

        if L < MIN_ROWS or L > MAX_ROWS:
            to_delete.append((path, L))
    except Exception as e:
        print(f"Error reading {path}: {e}")
        to_delete.append((path, -1))

print("\n=== SUMMARY ===")
print("Files found:", len(files))
print("Files to delete:", len(to_delete))
print(f"Delete %: {round(len(to_delete)/len(files)*100, 2)}%\n")

print("=== Files to delete (first 30) ===")
for p, L in to_delete[:30]:
    print(f"{L} rows -> {p}")

# DELETE CONFIRMATION
if DRY_RUN:
    print("\nDRY RUN mode: No files were deleted.")
else:
    print("\nDeleting files...")
    for p, _ in to_delete:
        try:
            os.remove(p)
        except Exception as e:
            print("Error deleting:", p, e)
    print("Done!")
