#!/usr/bin/env python3
"""
Converter: CSLDaily .pt files → Two-Stage SLG format
Reads individual .pt files from Data_CSLDaily/{train,dev,test}/
and produces .gloss, .skels, .files + metadata files.

Trims idle (no-movement) segments at start/end of each sequence
using the idle_times_*.jsonl files in segments_no_move/.
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def load_idle_times(jsonl_path: Path) -> dict:
    """
    Load idle segment info from a .jsonl file.
    Returns dict: video_name -> list of {start_time, end_time}
    """
    idle_map = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            idle_map[entry["video_path"]] = entry["idle_segments"]
    return idle_map


def get_active_range(idle_segments: list, num_frames: int):
    """
    Given idle segments, return (active_start, active_end) excluding
    idle periods at the beginning and end of the sequence.
    """
    active_start = 0
    active_end = num_frames
    for seg in idle_segments:
        if seg["start_time"] == 0:
            active_start = seg["end_time"] + 1
        if seg["end_time"] == num_frames - 1:
            active_end = seg["start_time"]
    return active_start, active_end


def convert_split(
    input_dir: Path,
    output_dir: Path,
    metadata_dir: Path,
    idle_dir: Path,
    split: str,
):
    """
    Convert one split (train/dev/test) of CSLDaily .pt files to two-stage format.

    Produces:
      output_dir/{split}.gloss   - one gloss line per sequence (space-separated tokens)
      output_dir/{split}.skels   - one line per sequence: 184 floats/frame flattened
      output_dir/{split}.files   - one filename per line
      metadata_dir/{split}_texts.txt   - name|text per line
      metadata_dir/{split}_signers.txt - name,signer per line
    """
    pt_dir = input_dir / split
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith(".pt")])

    if not pt_files:
        print(f"  No .pt files found in {pt_dir}, skipping.")
        return

    # Load idle times for this split
    idle_path = idle_dir / f"idle_times_{split}.jsonl"
    idle_map = load_idle_times(idle_path) if idle_path.exists() else {}
    if idle_map:
        print(f"  Loaded idle times for {len(idle_map)} sequences")
    else:
        print(f"  WARNING: No idle times found at {idle_path}, using full sequences")

    gloss_lines = []
    skels_lines = []
    files_lines = []
    texts_lines = []
    signers_lines = []

    skipped = 0

    for fname in tqdm(pt_files, desc=f"  {split}"):
        data = torch.load(pt_dir / fname, map_location="cpu", weights_only=False)

        poses = data["poses_3d_filtered"]  # [N_frames, 61, 3]
        gloss_tokens = data["gloss"]       # list of str
        text = data["text"]                # str
        name = data["name"]                # str
        signer = data.get("signer", name)  # str

        n_frames = poses.shape[0]

        # Trim idle segments at start/end
        if name in idle_map:
            active_start, active_end = get_active_range(idle_map[name], n_frames)
            poses = poses[active_start:active_end]
            n_frames = poses.shape[0]

        if n_frames == 0:
            skipped += 1
            continue

        # Flatten [N, 61, 3] -> [N, 183]
        flat = poses.reshape(n_frames, -1).numpy()  # [N, 183]

        # Append counter column: 0 -> 1 linearly
        counter = np.linspace(0.0, 1.0, n_frames).reshape(-1, 1)
        flat_with_counter = np.concatenate([flat, counter], axis=1)  # [N, 184]

        # Format as space-separated floats, all frames in one line
        skels_str = " ".join(f"{v:.6f}" for v in flat_with_counter.ravel())

        gloss_lines.append(" ".join(gloss_tokens))
        skels_lines.append(skels_str)
        files_lines.append(name)
        texts_lines.append(f"{name}|{text}")
        signers_lines.append(f"{name},{signer}")

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{split}.gloss").write_text("\n".join(gloss_lines) + "\n")
    (output_dir / f"{split}.skels").write_text("\n".join(skels_lines) + "\n")
    (output_dir / f"{split}.files").write_text("\n".join(files_lines) + "\n")
    (metadata_dir / f"{split}_texts.txt").write_text("\n".join(texts_lines) + "\n")
    (metadata_dir / f"{split}_signers.txt").write_text("\n".join(signers_lines) + "\n")

    print(f"  {split}: {len(gloss_lines)} sequences written, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSLDaily .pt files to Two-Stage SLG format"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Data_CSLDaily",
        help="Directory with train/dev/test subdirs of .pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Data" / "csldaily",
        help="Output directory for .gloss/.skels/.files",
    )
    parser.add_argument(
        "--metadata_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Data" / "metadata_csldaily",
        help="Output directory for metadata (texts, signers)",
    )
    parser.add_argument(
        "--idle_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "segments_no_move",
        help="Directory with idle_times_*.jsonl files",
    )
    args = parser.parse_args()

    print(f"Input:    {args.input_dir}")
    print(f"Output:   {args.output_dir}")
    print(f"Metadata: {args.metadata_dir}")
    print(f"Idle:     {args.idle_dir}")
    print()

    for split in ["train", "dev", "test"]:
        convert_split(args.input_dir, args.output_dir, args.metadata_dir, args.idle_dir, split)

    print("\nDone.")


if __name__ == "__main__":
    main()
