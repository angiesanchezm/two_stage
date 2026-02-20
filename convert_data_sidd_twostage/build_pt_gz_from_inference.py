#!/usr/bin/env python3
"""
Build dev.pt.gz and test.pt.gz from inference results.

Assembles SignJoey-format .pt.gz files from:
- Models/german/*_test_results_{split}/file_paths.txt  -> name
- Models/german/*_test_results_{split}/hypotheses.skels -> sign tensor
- Data/metadata/{split}_texts.txt                        -> text
- Data/metadata/{split}_signers.txt                      -> signer
- Data/german/{split}.files + {split}.gloss              -> gloss
"""

import argparse
import gzip
import joblib
import numpy as np
import torch
from pathlib import Path


def load_texts(metadata_dir: Path, split: str) -> dict:
    """Load name|text mapping from metadata."""
    texts = {}
    path = metadata_dir / f"{split}_texts.txt"
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                name, text = line.split('|', 1)
                texts[name] = text
    print(f"  Loaded {len(texts)} texts from {path}")
    return texts


def load_signers(metadata_dir: Path, split: str) -> dict:
    """Load name,signer mapping from metadata."""
    signers = {}
    path = metadata_dir / f"{split}_signers.txt"
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ',' in line:
                name, signer = line.split(',', 1)
                signers[name] = signer
    print(f"  Loaded {len(signers)} signers from {path}")
    return signers


def load_glosses(data_dir: Path, split: str) -> dict:
    """Load name->gloss mapping from .files + .gloss (aligned by line)."""
    glosses = {}
    files_path = data_dir / f"{split}.files"
    gloss_path = data_dir / f"{split}.gloss"

    with open(files_path, 'r', encoding='utf-8') as ff, \
         open(gloss_path, 'r', encoding='utf-8') as gf:
        for name_line, gloss_line in zip(ff, gf):
            name = name_line.strip()
            gloss = gloss_line.strip()
            glosses[name] = gloss

    print(f"  Loaded {len(glosses)} glosses from {gloss_path}")
    return glosses


def parse_skels_line(line: str, frame_dim: int = 184, joint_dim: int = 183) -> torch.Tensor:
    """Parse a single line of hypotheses.skels into a [N_frames, 183] tensor.

    Each line has N_frames * 184 floats (183 joint coords + 1 counter per frame).
    We drop the counter column.
    """
    values = np.array(line.strip().split(), dtype=np.float32)
    n_values = len(values)
    assert n_values % frame_dim == 0, \
        f"Number of values ({n_values}) is not a multiple of {frame_dim}"

    n_frames = n_values // frame_dim
    reshaped = values.reshape(n_frames, frame_dim)
    # Drop the last column (counter)
    joints_only = reshaped[:, :joint_dim]
    return torch.from_numpy(joints_only)


def build_pt_gz(results_dir: Path, data_dir: Path, metadata_dir: Path,
                split: str, output_path: Path):
    """Build a .pt.gz file for one split."""
    print(f"\n{'='*70}")
    print(f"Building {output_path.name} from {results_dir}")
    print(f"{'='*70}")

    # 1. Load names
    names_path = results_dir / "file_paths.txt"
    with open(names_path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"  {len(names)} sequences from {names_path}")

    # 2. Load hypotheses skels
    skels_path = results_dir / "hypotheses.skels"
    with open(skels_path, 'r', encoding='utf-8') as f:
        skel_lines = [line for line in f if line.strip()]
    assert len(skel_lines) == len(names), \
        f"Mismatch: {len(names)} names vs {len(skel_lines)} skel lines"

    # 3. Load lookup tables
    texts = load_texts(metadata_dir, split)
    signers = load_signers(metadata_dir, split)
    glosses = load_glosses(data_dir, split)

    # 4. Assemble samples
    samples = []
    missing = {'text': 0, 'signer': 0, 'gloss': 0}

    for i, name in enumerate(names):
        sign = parse_skels_line(skel_lines[i])

        text = texts.get(name)
        if text is None:
            missing['text'] += 1
            text = ''

        signer = signers.get(name)
        if signer is None:
            missing['signer'] += 1
            signer = 'unknown'

        gloss = glosses.get(name)
        if gloss is None:
            missing['gloss'] += 1
            gloss = ''

        samples.append({
            'name': name,
            'text': text,
            'gloss': gloss,
            'sign': sign,
            'signer': signer,
        })

    # Report missing
    for field, count in missing.items():
        if count > 0:
            print(f"  WARNING: {count} samples missing '{field}'")

    # 5. Stats
    frame_counts = [s['sign'].shape[0] for s in samples]
    print(f"\n  Samples: {len(samples)}")
    print(f"  Sign shape example: {samples[0]['sign'].shape}")
    print(f"  Frames - min: {min(frame_counts)}, max: {max(frame_counts)}, "
          f"mean: {np.mean(frame_counts):.1f}")

    # 6. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, 'wb') as f:
        joblib.dump(samples, f)
    print(f"\n  Saved {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Build .pt.gz files from inference results and metadata'
    )
    parser.add_argument('--models-dir', type=str,
                        default='../Models/german',
                        help='Directory containing *_test_results_{dev,test}/')
    parser.add_argument('--data-dir', type=str,
                        default='../Data/german',
                        help='Directory containing {split}.files and {split}.gloss')
    parser.add_argument('--metadata-dir', type=str,
                        default='../Data/metadata',
                        help='Directory containing {split}_texts.txt and {split}_signers.txt')
    parser.add_argument('--output-dir', type=str,
                        default='../Models/german',
                        help='Output directory for .pt.gz files')

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)

    for split in ['dev', 'test']:
        # Auto-detect folder matching *_test_results_{split}
        matches = sorted(models_dir.glob(f"*_test_results_{split}"))
        if not matches:
            print(f"Skipping {split}: no *_test_results_{split}/ found in {models_dir}")
            continue
        results_dir = matches[-1]  # use latest if multiple
        if len(matches) > 1:
            print(f"  Multiple results dirs for {split}, using: {results_dir.name}")

        output_path = output_dir / f"{split}.pt.gz"
        build_pt_gz(results_dir, data_dir, metadata_dir, split, output_path)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
