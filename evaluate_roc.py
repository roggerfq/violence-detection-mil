"""
Project: VD-MIL: A Deep Multiple Instance Learning Approach for Violence Detection in Surveillance Videos
Author: Roger Figueroa Quintero
Years: 2025–2026

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

This code is part of an academic/research project.
You are free to use, modify, and share this code for non-commercial purposes only,
provided that proper credit is given to the author.

Commercial use of this code is strictly prohibited without explicit written permission
from the author.

Full license text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from movinet_classifier import MovinetClassifier, Net
from activity_detection import run_video_processing
from signal_smoothing import smooth_scores

def get_video_files(folder_path):
    """Get all video files from a folder."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(folder_path, file))
    
    return sorted(video_files)

def evaluate_videos(video_paths, classifier, target_fps=8, length_window=8, stride=4):
    """Evaluate all videos and return their scores."""
    all_scores = []
    max_queue_size = 50  # Default value
    
    for i, video_path in enumerate(video_paths):
        print(f"Processing video {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
        try:
            scores = run_video_processing(video_path, classifier, target_fps, length_window, stride, max_queue_size)
            scores_smooth = smooth_scores(scores)
            all_scores.extend(scores_smooth)  # Flatten scores from all segments
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue
    
    return np.array(all_scores)

def plot_roc_curve(scores_positive, scores_negative, output_path):
    """Generate and save ROC curve."""
    # Create labels: 1 for positive, 0 for negative
    y_true = np.concatenate([
        np.ones(len(scores_positive)),
        np.zeros(len(scores_negative))
    ])
    
    # Combine all scores
    y_scores = np.concatenate([scores_positive, scores_negative])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to: {output_path}")
    print(f"AUC Score: {roc_auc:.4f}")
    
    plt.close()
    
    return roc_auc, fpr, tpr, thresholds

def main():
    parser = argparse.ArgumentParser(description='Evaluate video classifier and generate ROC curve')
    parser.add_argument('--path_model', type=str, required=True,
                        help='Path to the .pt PyTorch model file')
    parser.add_argument('--folder_backbone_model', type=str, required=True,
                        help='Path to folder containing backbone model weights')
    parser.add_argument('--path_positive_set', type=str, required=True,
                        help='Path to folder containing positive video samples')
    parser.add_argument('--path_negative_set', type=str, required=True,
                        help='Path to folder containing negative video samples')
    parser.add_argument('--path_roc_image', type=str, required=True,
                        help='Output path for ROC curve image (PNG)')
    parser.add_argument('--target_fps', type=int, default=8,
                        help='Target FPS for video processing (default: 8)')
    parser.add_argument('--length_window', type=int, default=8,
                        help='Length of window in frames (default: 8)')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride for video processing (default: 4)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for inference (default: cpu)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.path_model):
        raise FileNotFoundError(f"Model file not found: {args.path_model}")
    if not os.path.exists(args.folder_backbone_model):
        raise FileNotFoundError(f"Backbone model folder not found: {args.folder_backbone_model}")
    if not os.path.exists(args.path_positive_set):
        raise FileNotFoundError(f"Positive set folder not found: {args.path_positive_set}")
    if not os.path.exists(args.path_negative_set):
        raise FileNotFoundError(f"Negative set folder not found: {args.path_negative_set}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.path_roc_image), exist_ok=True)
    
    print("="*60)
    print("Video Classification ROC Evaluation")
    print("="*60)
    print(f"Model: {args.path_model}")
    print(f"Backbone: {args.folder_backbone_model}")
    print(f"Device: {args.device}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Length Window: {args.length_window}")
    print(f"Stride: {args.stride}")
    print(f"Positive set: {args.path_positive_set}")
    print(f"Negative set: {args.path_negative_set}")
    print("="*60)
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = MovinetClassifier(
        args.folder_backbone_model, 
        [args.path_model], 
        device=args.device
    )
    
    # Get video files
    positive_videos = get_video_files(args.path_positive_set)
    negative_videos = get_video_files(args.path_negative_set)
    
    print(f"\nFound {len(positive_videos)} positive videos")
    print(f"Found {len(negative_videos)} negative videos")
    
    if len(positive_videos) == 0 or len(negative_videos) == 0:
        raise ValueError("Both positive and negative sets must contain at least one video")
    
    # Evaluate positive set
    print("\n" + "="*60)
    print("Evaluating POSITIVE videos...")
    print("="*60)
    scores_positive_set = evaluate_videos(
        positive_videos, 
        classifier, 
        args.target_fps, 
        args.length_window, 
        args.stride
    )
    
    # Evaluate negative set
    print("\n" + "="*60)
    print("Evaluating NEGATIVE videos...")
    print("="*60)
    scores_negative_set = evaluate_videos(
        negative_videos, 
        classifier, 
        args.target_fps, 
        args.length_window, 
        args.stride
    )
    
    # Generate ROC curve
    print("\n" + "="*60)
    print("Generating ROC curve...")
    print("="*60)
    print(f"Total positive segments: {len(scores_positive_set)}")
    print(f"Total negative segments: {len(scores_negative_set)}")
    
    roc_auc, fpr, tpr, thresholds = plot_roc_curve(
        scores_positive_set, 
        scores_negative_set, 
        args.path_roc_image
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"Positive scores - Mean: {np.mean(scores_positive_set):.4f}, "
          f"Std: {np.std(scores_positive_set):.4f}")
    print(f"Negative scores - Mean: {np.mean(scores_negative_set):.4f}, "
          f"Std: {np.std(scores_negative_set):.4f}")
    print("="*60)
    print("Evaluation completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
