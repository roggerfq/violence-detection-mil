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
