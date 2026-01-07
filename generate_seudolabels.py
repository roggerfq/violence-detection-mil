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
import sys
from pathlib import Path

# Get path and label from command line arguments
if len(sys.argv) < 3:
    print("Error: Please provide video path and label as arguments")
    print("Usage: python script.py <video_path> <label>")
    print("Example: python script.py /content/dataset/Violence 0,inf")
    print("Example: python script.py /content/dataset/Violence 0,4")
    print("Example: python script.py /content/dataset/Violence -1,-1")
    sys.exit(1)

video_path = sys.argv[1]
label = sys.argv[2]

# Get all .avi and .mp4 files
video_files = []
for ext in ['*.avi', '*.mp4']:
    video_files.extend(Path(video_path).glob(ext))

# Sort files for consistent ordering
video_files.sort()

# Create labels.txt content
labels_content = []
for video_file in video_files:
    labels_content.append(f"{video_file.name} {label}\n")

# Write to labels.txt in the same directory
output_path = os.path.join(video_path, "labels.txt")
with open(output_path, 'w') as f:
    f.writelines(labels_content)

print(f"✓ Found {len(video_files)} video files")
print(f"✓ labels.txt created at: {output_path}")
print(f"✓ Using label: {label}")
print("\nFirst 5 entries:")
for line in labels_content[:5]:
    print(f"  {line.strip()}")
    