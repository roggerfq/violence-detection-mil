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

fps = sys.argv[1]
path_videos = sys.argv[2]
path_new_videos = sys.argv[3]

#path_videos = './videos'
#path_new_videos = './videos_8fps'

os.system("mkdir -p " + path_new_videos)

for f in os.listdir(path_videos):
    if f.endswith(".mp4") or f.endswith(".avi"):
        video_file = os.path.join(path_videos, f)
        new_video_file = os.path.join(path_new_videos, f)
        cmd = 'ffmpeg -i ' + video_file + ' -r '+ str(fps) + ' -y ' + new_video_file
        print(cmd)
        os.system(cmd)
