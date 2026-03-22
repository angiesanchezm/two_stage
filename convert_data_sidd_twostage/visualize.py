import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import torch
import joblib
import gzip
from os.path import join as pjoin


# ============================================================================
# MediaPipe Skeleton Connections
# ============================================================================
BODY_CONNECTIONS = [
    (0, 1), (1, 2),  # Right arm
    (3, 4), (4, 5),  # Left arm
    (0, 3),  # Shoulders
    (0, 6), (3, 7), (6, 7),  # Torso
    (8, 0), (8, 3),  # Face to body
    (8, 9), (9, 10), (10, 11), (11, 15),  # Face left
    (8, 12), (12, 13), (13, 14), (14, 16),  # Face right
    (17, 18),  # Mouth
]

RIGHT_HAND_CONNECTIONS = [
    (2, 19),  # Wrist connection
    (19, 20), (20, 21), (21, 22), (22, 23),  # Thumb
    (19, 24), (24, 25), (25, 26), (26, 27),  # Index
    (19, 28), (28, 29), (29, 30), (30, 31),  # Middle
    (19, 32), (32, 33), (33, 34), (34, 35),  # Ring
    (19, 36), (36, 37), (37, 38), (38, 39),  # Pinky
    (24, 28), (28, 32), (32, 36),  # Palm
]

LEFT_HAND_CONNECTIONS = [
    (5, 40),  # Wrist connection
    (40, 41), (41, 42), (42, 43), (43, 44),  # Thumb
    (40, 45), (45, 46), (46, 47), (47, 48),  # Index
    (40, 49), (49, 50), (50, 51), (51, 52),  # Middle
    (40, 53), (53, 54), (54, 55), (55, 56),  # Ring
    (40, 57), (57, 58), (58, 59), (59, 60),  # Pinky
    (45, 49), (49, 53), (53, 57),  # Palm
]

ALL_CONNECTIONS = BODY_CONNECTIONS + RIGHT_HAND_CONNECTIONS + LEFT_HAND_CONNECTIONS


def plot_skeleton_frame(ax, keypoints, color, title, frame_num, total_frames):
    """Plot 3D skeleton with proper connections."""
    ax.clear()

    valid_mask = ~np.all(np.isclose(keypoints, 0), axis=1)

    sizes = np.ones(len(keypoints)) * 20
    sizes[:19] = 50  # Larger for body

    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
              c=color, s=sizes, alpha=0.8, edgecolors='white', linewidth=0.5)

    for connection in ALL_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            if valid_mask[idx1] and valid_mask[idx2]:
                point1 = keypoints[idx1]
                point2 = keypoints[idx2]
                linewidth = 3.5 if (idx1 < 19 and idx2 < 19) else 1.8
                alpha = 0.9 if (idx1 < 19 and idx2 < 19) else 0.7
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]],
                       c=color, linewidth=linewidth, alpha=alpha)

    ax.set_xlabel('X', fontsize=10, labelpad=8)
    ax.set_ylabel('Y', fontsize=10, labelpad=8)
    ax.set_zlabel('Z', fontsize=10, labelpad=8)
    ax.set_title(f'{title}\nFrame {frame_num+1}/{total_frames}',
                fontsize=12, fontweight='bold', pad=10)

    valid_pts = keypoints[valid_mask]
    if len(valid_pts) > 0:
        max_range = np.array([
            valid_pts[:, 0].max() - valid_pts[:, 0].min(),
            valid_pts[:, 1].max() - valid_pts[:, 1].min(),
            valid_pts[:, 2].max() - valid_pts[:, 2].min()
        ]).max() / 2.0

        mid_x = (valid_pts[:, 0].max() + valid_pts[:, 0].min()) * 0.5
        mid_y = (valid_pts[:, 1].max() + valid_pts[:, 1].min()) * 0.5
        mid_z = (valid_pts[:, 2].max() + valid_pts[:, 2].min()) * 0.5

        margin = max_range * 0.2
        ax.set_xlim(mid_x - max_range - margin, mid_x + max_range + margin)
        ax.set_ylim(mid_y - max_range - margin, mid_y + max_range + margin)
        ax.set_zlim(mid_z - max_range - margin, mid_z + max_range + margin)

    ax.view_init(elev=-80, azim=-90)
    ax.grid(True, alpha=0.3)


def load_predictions(predictions_file):
    """Load predictions from .pt.gz file (joblib or torch format)."""
    print(f"\n{'='*60}")
    print(f"Loading predictions from: {predictions_file}")

    try:
        with gzip.open(predictions_file, 'rb') as f:
            predictions_list = torch.load(f, map_location='cpu', weights_only=False)
    except Exception:
        with gzip.open(predictions_file, 'rb') as f:
            predictions_list = joblib.load(f)

    print(f"Loaded {len(predictions_list)} predictions")
    print(f"{'='*60}\n")

    return predictions_list


def load_ground_truth_from_skels(data_dir, split):
    """Load all ground truth sequences from Data/german/{split}.skels + .files.

    Returns dict mapping name -> numpy array of shape [N_frames, 61, 3].
    """
    files_path = pjoin(data_dir, f"{split}.files")
    skels_path = pjoin(data_dir, f"{split}.skels")

    if not os.path.exists(files_path) or not os.path.exists(skels_path):
        print(f"Warning: GT files not found: {files_path} / {skels_path}")
        return {}

    print(f"Loading ground truth from {skels_path}...")
    gt_dict = {}
    with open(files_path, 'r') as ff, open(skels_path, 'r') as sf:
        for name_line, skel_line in zip(ff, sf):
            name = name_line.strip()
            values = np.array(skel_line.strip().split(), dtype=np.float32)
            n_values = len(values)
            # .skels has 184 floats per frame (183 joints + 1 counter)
            if n_values % 184 == 0:
                n_frames = n_values // 184
                reshaped = values.reshape(n_frames, 184)
                joints = reshaped[:, :183].reshape(n_frames, 61, 3)
            elif n_values % 183 == 0:
                n_frames = n_values // 183
                joints = values.reshape(n_frames, 61, 3)
            else:
                continue
            gt_dict[name] = joints

    print(f"Loaded {len(gt_dict)} ground truth sequences")
    return gt_dict


def to_3d_keypoints(sign_data):
    """Convert sign tensor/array [N, 183] to [N, 61, 3]."""
    if isinstance(sign_data, torch.Tensor):
        sign_data = sign_data.numpy()
    if sign_data.ndim == 2 and sign_data.shape[1] == 183:
        return sign_data.reshape(-1, 61, 3)
    return sign_data


def play_comparison_interactive(predictions_file, sample_idx=0, data_dir=None, split='dev'):
    """
    Interactive visualization comparing prediction with ground truth.
    If data_dir is not provided, shows prediction only.

    Controls:
    - Right arrow / 'd': Next frame
    - Left arrow / 'a': Previous frame
    - Space: Play/Pause
    - 'n': Next sample
    - 'p': Previous sample
    - 'q': Quit
    """
    predictions = load_predictions(predictions_file)

    # Load ground truth if data_dir provided
    gt_dict = {}
    if data_dir and os.path.isdir(data_dir):
        gt_dict = load_ground_truth_from_skels(data_dir, split)

    has_gt = len(gt_dict) > 0

    if sample_idx >= len(predictions):
        print(f"Error: Sample index {sample_idx} out of range (max: {len(predictions) - 1})")
        return

    pred_dict = predictions[sample_idx]

    print(f"{'='*60}")
    print(f"Sample {sample_idx + 1}/{len(predictions)}:")
    print(f"  Name: {pred_dict['name']}")
    print(f"  Signer: {pred_dict.get('signer', 'N/A')}")
    print(f"  Gloss: {pred_dict.get('gloss', 'N/A')}")
    print(f"  Text: {pred_dict.get('text', 'N/A')}")
    print(f"  Prediction shape: {pred_dict['sign'].shape}")
    if has_gt and pred_dict['name'] in gt_dict:
        gt = gt_dict[pred_dict['name']]
        print(f"  Ground truth: {gt.shape[0]} frames")
    elif has_gt:
        print(f"  Ground truth: not found for this sample")
    else:
        print(f"  Ground truth: not loaded (prediction-only mode)")
    print(f"{'='*60}\n")

    print(f"{'='*60}")
    print("CONTROLS:")
    print("  Mouse: Rotate views")
    print("  Right arrow / 'd': Next frame")
    print("  Left arrow / 'a': Previous frame")
    print("  Space: Play/Pause")
    print("  'n': Next sample")
    print("  'p': Previous sample")
    print("  'q': Quit")
    print(f"{'='*60}\n")

    plt.style.use('dark_background')

    if has_gt:
        fig = plt.figure(figsize=(22, 7))
    else:
        fig = plt.figure(figsize=(11, 7))
    fig.patch.set_facecolor('#1a1a1a')

    if has_gt:
        ax1 = fig.add_subplot(121, projection='3d', facecolor='#1a1a1a')
        ax2 = fig.add_subplot(122, projection='3d', facecolor='#1a1a1a')
    else:
        ax2 = fig.add_subplot(111, projection='3d', facecolor='#1a1a1a')
        ax1 = None

    state = {
        'frame': 0,
        'playing': False,
        'last_update': time.time(),
        'sample_idx': sample_idx,
        'predictions': predictions,
        'gt_dict': gt_dict,
        'has_gt': has_gt,
    }

    def get_motions():
        curr_pred = state['predictions'][state['sample_idx']]
        pred_3d = to_3d_keypoints(curr_pred['sign'])
        gt_3d = state['gt_dict'].get(curr_pred['name']) if state['has_gt'] else None
        return pred_3d, gt_3d, curr_pred

    def get_max_frames():
        pred_3d, gt_3d, _ = get_motions()
        if gt_3d is not None:
            return max(len(pred_3d), len(gt_3d))
        return len(pred_3d)

    def update_plot():
        frame = state['frame']
        pred_3d, gt_3d, curr_pred = get_motions()

        title_text = (f"Sample {state['sample_idx'] + 1}/{len(state['predictions'])} | "
                     f"Name: {curr_pred['name']}\n"
                     f"Gloss: {curr_pred.get('gloss', 'N/A')}\n"
                     f"Text: {curr_pred.get('text', 'N/A')}")
        fig.suptitle(title_text, fontsize=14, fontweight='bold', color='white')

        if state['has_gt'] and gt_3d is not None:
            gt_frame = min(frame, len(gt_3d) - 1)
            plot_skeleton_frame(ax1, gt_3d[gt_frame], '#00ff41',
                              f'Ground Truth ({len(gt_3d)}f)', gt_frame, len(gt_3d))
        elif ax1 is not None:
            ax1.clear()
            ax1.set_title('Ground Truth\n(not available)', fontsize=12, fontweight='bold')

        pred_frame = min(frame, len(pred_3d) - 1)
        plot_skeleton_frame(ax2, pred_3d[pred_frame], '#00d9ff',
                          f'Prediction ({len(pred_3d)}f)', pred_frame, len(pred_3d))

        plt.tight_layout(rect=[0, 0.02, 1, 0.88])
        fig.canvas.draw_idle()

    def change_sample(new_idx):
        if 0 <= new_idx < len(state['predictions']):
            state['sample_idx'] = new_idx
            state['frame'] = 0

            pred = state['predictions'][new_idx]
            print(f"\n{'='*60}")
            print(f"Sample {new_idx + 1}/{len(state['predictions'])}:")
            print(f"  Name: {pred['name']}")
            print(f"  Gloss: {pred.get('gloss', 'N/A')}")
            print(f"  Text: {pred.get('text', 'N/A')}")
            print(f"{'='*60}\n")

            update_plot()

    def on_key(event):
        if event.key in ['right', 'd']:
            max_f = get_max_frames()
            state['frame'] = (state['frame'] + 1) % max_f
            update_plot()
        elif event.key in ['left', 'a']:
            max_f = get_max_frames()
            state['frame'] = (state['frame'] - 1) % max_f
            update_plot()
        elif event.key == ' ':
            state['playing'] = not state['playing']
            print("Playing" if state['playing'] else "Paused")
        elif event.key == 'n':
            change_sample(state['sample_idx'] + 1)
        elif event.key == 'p':
            change_sample(state['sample_idx'] - 1)
        elif event.key == 'q':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)

    update_plot()

    try:
        plt.show(block=False)
        while plt.fignum_exists(fig.number):
            if state['playing']:
                current_time = time.time()
                if current_time - state['last_update'] >= 1 / 15:
                    max_f = get_max_frames()
                    state['frame'] = (state['frame'] + 1) % max_f
                    update_plot()
                    state['last_update'] = current_time
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        plt.close()
        print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize predictions from .pt.gz files')
    parser.add_argument('--predictions_file', type=str, required=True,
                       help='Path to .pt.gz predictions file')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize (0-based)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory with {split}.skels and {split}.files for GT comparison')
    parser.add_argument('--split', type=str, default='dev',
                       choices=['train', 'dev', 'test'],
                       help='Data split for ground truth')

    args = parser.parse_args()

    play_comparison_interactive(
        args.predictions_file,
        args.sample_idx,
        args.data_dir,
        args.split
    )




# Usage examples:                                                                                         

#   Prediction only:
#   python visualize.py --predictions_file Models_100/german/dev.pt.gz
#   python visualize.py --predictions_file Models_100/german/dev.pt.gz --sample_idx 200
#   With ground truth comparison:
#   python visualize.py --predictions_file Models_100/german/dev.pt.gz --data_dir Data/german --split dev
#.  python visualize.py --predictions_file Models_100/german/dev.pt.gz --sample_idx 200 --data_dir Data/german --split dev
#   For 300 model:
#   python visualize.py --predictions_file Models_300/german/dev.pt.gz --data_dir Data/german --split dev
#   python visualize.py --predictions_file Models_300/german/dev.pt.gz --sample_idx 200 --data_dir Data/german --split dev

#ahora sí, haz configurable el max_length y test_data/dev_data desde el yaml

# python convert_data_sidd_twostage/visualize.py \                                                                                                                
#     --predictions_file ./Models/csldaily/test.pt.gz \                                                                                                             
#     --sample_idx 200 \                                                                                                                                            
#     --data_dir ./Data/csldaily \                                                                                                                                  
#     --split test 