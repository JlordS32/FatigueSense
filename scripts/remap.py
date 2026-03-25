"""
Remaps YOLO label class IDs in vid_001/labels from:
  0: Mouth, 1: Torso, 2: Eyes, 3: Head
to:
  0: Head,  1: Torso, 2: Mouth, 3: Eyes
"""

import os
from pathlib import Path

# old_id -> new_id
REMAP = {0: 2, 1: 1, 2: 3, 3: 0}

LABELS_DIR = Path(__file__).parent.parent / "FatigueSense" / "dataset_" / "vid_001" / "labels" / "train"


def remap_file(path: Path) -> int:
    lines = path.read_text().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_id = int(parts[0])
        new_id = REMAP.get(old_id, old_id)
        new_lines.append(" ".join([str(new_id)] + parts[1:]))
    path.write_text("\n".join(new_lines) + "\n")

    return len(new_lines)


def main():
    txt_files = list(LABELS_DIR.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under {LABELS_DIR}")
        return

    for f in txt_files:
        count = remap_file(f)
        print(
            f"Remapped {count} annotations in {f.relative_to(LABELS_DIR.parent.parent)}"
        )

    print(f"\nDone — {len(txt_files)} file(s) updated.")


if __name__ == "__main__":
    main()