import os
from collections import defaultdict

DATA_DIR = "data"   # change if needed

def count_zone_files(data_dir=DATA_DIR):
    zone_counts = defaultdict(int)
    total_files = 0
    csv_files = []

    # scan directory
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            csv_files.append(fname)
            total_files += 1

            # extract zone number from filename: before first underscore
            try:
                zone = fname.split("_")[0]
                zone_counts[zone] += 1
            except:
                print(f"Could not parse zone from: {fname}")

    # print summary
    print("\n=== Zone File Count Summary ===")
    print(f"Total CSV files: {total_files}\n")

    for zone in sorted(zone_counts.keys(), key=lambda x: int(x)):
        print(f"Zone {zone}: {zone_counts[zone]} files")

    # show any missing zones
    print("\n=== Missing Zones (0–9) ===")
    for z in range(10):
        if str(z) not in zone_counts:
            print(f"Zone {z}: 0 files")

    print("\nDone.")

if __name__ == "__main__":
    count_zone_files()
