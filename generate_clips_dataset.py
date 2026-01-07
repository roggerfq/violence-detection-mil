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
import cv2
import random
from utils import Sampler


'''
def read_labels(label_file):
    label_info = []
    with open(label_file, 'r') as file_in:
         for line in file_in:
             info = {}
             line = line.strip() #remove '\n' in line
             aux1 = line.split(' ')

             name_video = aux1[0]
             type_cam = int(aux1[1])
             info['name_video'] = name_video
             info['type_cam'] = type_cam

             info['gt'] = []
             if(len(aux1) > 2):
                aux2 = aux1[2].split(',')
                gt = (len(aux2)//2)*[None]
                for i in range(len(aux2)//2):
                    gt[i] = (int(aux2[2*i]), int(aux2[2*i+1]))
                info['gt'] = gt

             if(len(aux1) > 3):
                type_activity = aux1[3]
                info['type_activity'] = type_activity
            
             label_info.append(info)

    return label_info
'''               


def read_labels(label_file):
    label_info = []
    with open(label_file, 'r') as file_in:
         for line in file_in:
             info = {}
             line = line.strip() #remove '\n' in line
             aux1 = line.split(' ')

             name_video = aux1[0]
             info['name_video'] = name_video

             info['gt'] = []
             if(len(aux1) > 1):
                aux2 = aux1[1].split(',')
                gt = (len(aux2)//2)*[None]
                for i in range(len(aux2)//2):
                    if(aux2[2*i+1] == "inf"):
                       aux2[2*i+1] = 10**1000 #all frames
                    gt[i] = (int(aux2[2*i]), int(aux2[2*i+1]))
                info['gt'] = gt
             label_info.append(info)

    return label_info



def get_number_frames(file_video, new_fps):
    '''
    by default only the frames resulting from the subsampling are returned,
    therefore flag_proc is always true
    '''
    cap = Sampler(file_video, new_fps)

    # sampling videos with fps < new_fps is not allowed
    if cap.original_fps < new_fps:
       raise AssertionError('The fps of ' + file_video + ' is less than new_fps = ' + str(new_fps))

    cap.original_fps

    frame_count = 0
    while(True):
        ret, flag_proc, frame = cap.read()
        if(ret):
           frame_count = frame_count + 1
        else:
           break
    cap.release()
    return frame_count



def get_overlap(gt_frames, window): 
    '''
    inputs:
    window = (frame_s, frame_e)
    gt_frames = [(ti1, tf1), .....(tin, tfn)]

    This function computes the overlap between the ground truth 
	and the temporal window, relative to the length of the window:
    iou = (gt_frame ∩ window) / |window|
    iou = (([ti1, tf1] ∩ window)/ |window|) + ... + (([tin, tfn] ∩ window)/ |window|)
    '''
    iou = 0.0
    area_ref = max(0.0, window[1] - window[0] + 1.0)
    for gt in gt_frames:
        inter = (max(window[0], gt[0]), min(window[1], gt[1]))
        area_inter = max(0.0, inter[1] - inter[0] + 1.0)
        iou = iou + (area_inter/area_ref)
    return iou








parser = argparse.ArgumentParser()
parser.add_argument("--path_set_videos", type=str, required = True)
parser.add_argument("--path_clips", type=str, required = True)
parser.add_argument("--label_file", type=str, required = True)
parser.add_argument("--new_fps", type=int, default = 5)
parser.add_argument("--length_clip", type=int, default = 12)
parser.add_argument("--stride_window_clip", type=int, default = 6)
parser.add_argument("--overlap_positive_clip", type=float, default = 0.3)
parser.add_argument("--n_positive_clips", type=int, default = -1)
parser.add_argument("--n_negative_clips", type=int, default = -1)
args = parser.parse_args()




path_set_videos = args.path_set_videos
path_clips = args.path_clips
label_file = args.label_file
label_info = read_labels(label_file)

#new fps for all videos
new_fps = args.new_fps
length_clip = args.length_clip #en segundos
length_frames = new_fps*length_clip
stride_window_clip = args.stride_window_clip #en segundos
stride_frames = new_fps*stride_window_clip
overlap_positive_clip = args.overlap_positive_clip

n_positive_clips = args.n_positive_clips
n_negative_clips = args.n_negative_clips


#list of clips
#[(frame_s1, frame_e1).....(frame_sn, frame_en)]
positive_clips = []
negative_clips = [] 
for info in label_info:
    name_video = info['name_video']
    gt = info['gt']
    frame_count = get_number_frames(os.path.join(path_set_videos, name_video), new_fps)
    if(frame_count < length_frames):
       continue

    gt_frames = []
    if(len(gt) > 0):
       #gt_frames = [(ti1, tf1), .....(tin, tfn)]
       #ti <= frame_count - 1 and tf <= frame_count - 1
       gt_frames = [tuple((min(t*new_fps, frame_count - 1) for t in w)) for w in gt]


    '''
    At this point it has been verified that the video has frame_count frames when it is subsampled to
    the new_fps rate. On the other hand, the time range of any window of gt_frames is contained in [0, frame_count-1].
    '''
    for i in range(0, frame_count, stride_frames):# 0 <= i <= frame_count - 1
        frame_s = i # 0 <= frame_s <= frame_count - 1
        # length_frames = frame_e - frame_s + 1 -> frame_e = frame_s + length_frames - 1
        frame_e = frame_s + length_frames - 1 
         
        
        if(frame_e < frame_count): # 0 <= frame_s <= frame_e <= frame_count - 1
           #print(i, ' ', frame_s, ' ', frame_e, ' ', length_frames)
           window = (frame_s, frame_e) # the window always holds that window ⊆ [0, frame_count - 1]
           overlap = get_overlap(gt_frames, window)
           if(overlap >= overlap_positive_clip):
               positive_clips.append({'name_video': name_video, 'window': window, 'label': 1})  
           elif(overlap == 0):
               #negative_clips.append({'name_video': name_video, 'window': window, 'label': -1})  
               w1 = max(window[0] - 1, 0)
               w2 = min(window[-1] + 1, frame_count - 1)
               window_and_padding = (w1, w2)
               overlap = get_overlap(gt_frames, window_and_padding)
               if(overlap == 0):
                  negative_clips.append({'name_video': name_video, 'window': window, 'label': -1})  
        else:
           #######solving last segment######
           if(frame_e > frame_count - 1):
              frame_e = frame_count - 1
              frame_s = frame_e - length_frames + 1
           #################################
           #print(i, ' ', frame_s, ' ', frame_e, ' ', length_frames)
           window = (frame_s, frame_e) # the window always holds that window ⊆ [0, frame_count - 1]
           overlap = get_overlap(gt_frames, window)
           if(overlap >= overlap_positive_clip):
               positive_clips.append({'name_video': name_video, 'window': window, 'label': 1})  
           elif(overlap == 0):
               #negative_clips.append({'name_video': name_video, 'window': window, 'label': -1})  
               w1 = max(window[0] - 1, 0)
               w2 = min(window[-1] + 1, frame_count - 1)
               window_and_padding = (w1, w2)
               overlap = get_overlap(gt_frames, window_and_padding)
               if(overlap == 0):
                  negative_clips.append({'name_video': name_video, 'window': window, 'label': -1})  

           break
        




##randomizing
random.shuffle(positive_clips)
random.shuffle(negative_clips)
if(n_positive_clips > 0):
   positive_clips = positive_clips[0:n_positive_clips]
if(n_negative_clips > 0):
   negative_clips = negative_clips[0:n_negative_clips]

#writing clips
path_positive_clips = os.path.join(path_clips, 'positive')
path_negative_clips = os.path.join(path_clips, 'negative')
if not os.path.exists(path_positive_clips):
   os.makedirs(path_positive_clips)
if not os.path.exists(path_negative_clips):
   os.makedirs(path_negative_clips)


# joining clips by name_video
all_clips = (positive_clips + negative_clips)
name_videos = set([clip['name_video'] for clip in all_clips])
dict_clips = {name_video: [] for name_video in name_videos}

for clip in all_clips:
    name_video = clip['name_video']
    window = clip['window']
    label = clip['label']
    dict_clips[name_video].append({'window': window, 'label': label, 'out': None})


#write clips
for name_video in dict_clips:
  try:
    #Note: window is zero position based
    print(name_video)
    file_video = os.path.join(path_set_videos, name_video)
    cap = Sampler(file_video, new_fps)
    width = int(cap.get_width())
    height = int(cap.get_height())
    image_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     
    list_clips = dict_clips[name_video]
    frame_count = 0
    while(True):
        ret, flag_proc, frame = cap.read()
        if(ret):
           ###############################
           for clip in list_clips:
               window = clip['window']
               out = clip['out']

    
               if(frame_count == window[0]):
                  #########################
                  label = clip['label']
                  name_clip = name_video[0:-4] + '_w_' + str(int(window[0]//new_fps)) + '_' + str(int(window[1]//new_fps)) + '.mp4'
                  if(label == 1):
                     file_clip = os.path.join(path_positive_clips, name_clip)
                  elif(label == -1):
                     file_clip = os.path.join(path_negative_clips, name_clip)
                  out = cv2.VideoWriter(file_clip, fourcc = fourcc, fps = new_fps, frameSize = image_size)
                  clip['out'] = out

                  out.write(frame)
                  #########################
               elif((frame_count > window[0]) and (frame_count < window[1])):
                  out.write(frame)
               elif(frame_count == window[1]):
                  out.write(frame)
                  out.release()
           ###############################
           frame_count = frame_count + 1
        else:
           break
    
    cap.release()
  except Exception as e:
    print("An error occurred:", e)

### making sure that the clips have the correct number of frames ###
print('checking number of frames of each clip')
for path in [path_positive_clips, path_negative_clips]:
   for root, dirs, files in os.walk(path):
      for f in files:
        if f.endswith('.mp4'):
           file_clip = os.path.join(root, f)
           cap = cv2.VideoCapture(file_clip)
           frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
           fps = cap.get(cv2.CAP_PROP_FPS)
           cap.release()

           print(f, ' length_frames=', length_frames, ' cv2.CAP_PROP_FRAME_COUNT=', frame_count)
           if frame_count != length_frames:
              raise AssertionError(f + ' has an error in cv2.CAP_PROP_FRAME_COUNT')

           if fps != new_fps:
              raise AssertionError(f + ' has an error in cv2.CAP_PROP_FPS')
           print(f, ' new_fps=', new_fps, ' cv2.CAP_PROP_FPS=', fps)
