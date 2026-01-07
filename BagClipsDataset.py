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
import glob
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import random


#############UTILS###################
def get_list_files(path, list_ext):
    '''
    returns a list with the names of the files
    whose extension is contained in list_ext
    '''
    list_files = []
    for ext in list_ext:
        list_files =  list_files + glob.glob1(path, ext)
    return list_files

#####################################



################TOOLS################

def imshow_bags(data):
    import matplotlib.pyplot as plt
    x = data[0]
    y = data[1]
    print('shape bags: ', x.shape)
    print('shape labels: ', y.shape)
    for i in range(x.shape[1]):
        for j in range(x.shape[3]):
            img = x[0,i,:,j,:,:].permute(1,2,0)
            img = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB)
            if(y[0,i] == 1):
               cv2.imshow('Positive bag', img)
            else:
               cv2.imshow('Negative bag', img)
            cv2.waitKey(0)

    cv2.destroyWindow('Positive bag')
    cv2.destroyWindow('Negative bag')

#####################################



def sample_clip(file_video, length_window, stride, new_size = None):
    # a video with constant and integer fps is assumed
    cap = cv2.VideoCapture(file_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    # reading video clip
    arr_clip = None
    if(new_size is not None):
      #height = size[0], width = size[1]
      arr_clip = torch.zeros((frame_count, new_size[0], new_size[1], 3), dtype=torch.uint8)
    else:
      arr_clip = torch.zeros((frame_count, height, width, 3), dtype=torch.uint8)

    n_frame = 0
    while(True):
         ret, frame = cap.read()
         if(ret):
            # movinet requires RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ##########################
            if(new_size is not None):
              #height = size[0], width = size[1]
              frame = cv2.resize(frame, (new_size[1], new_size[0]), interpolation = cv2.INTER_AREA)
            ##########################

            arr_clip[n_frame] = torch.from_numpy(frame)
            n_frame = n_frame + 1
         else:
            break
     
    if(n_frame != frame_count):
       raise AssertionError(file_video + ' has an error in cv2.CAP_PROP_FRAME_COUNT')

    # extracting samples (windowing)
    n_windows = int((frame_count - length_window)/stride) + 1
    n_frames_rest = frame_count - ((n_windows-1)*stride + length_window)
    samples = n_windows*[None]

    frame_e = frame_count - 1
    for i in range(len(samples)):
        # window size in frames is frame_e - frame_s + 1 = length_window
        frame_s = stride*i # frame start
        frame_e = frame_s + length_window - 1 # frame end
        #print(frame_s, ' ', frame_e)
        samples[i] = arr_clip[None, frame_s:frame_e + 1] # remember slicing is not inclusive on the right

    
    # if the remaining frames to window are greater or equal to length_window/2
    # the last possible frame is frame_count - 1
    # the first non-sampled frame is frame_e + 1
    if(n_frames_rest >= (length_window//2)):
       #frame_e = frame_count - 1 -> frame_e - frame_s + 1 = length_window
       #frame_s = frame_e + 1 - length_window = frame_count - length_window
       frame_e = frame_count - 1
       frame_s = frame_e + 1 - length_window
       samples.append(arr_clip[None, frame_s:frame_e + 1])
       
  

    # all extracted samples from the clip must have a length in frames equal to length_window
    for sample in samples:
        if sample.shape[1] != length_window:
           raise AssertionError('error sample.shape[1] != length_window')
           
 
    samples = torch.cat(samples, dim=0)
    samples = samples.permute(0,4,1,2,3)
    return samples
      


class RandomCropVideo():
      def __init__(self, size):
          self.size = size

      def __call__(self, x):
          height = x.shape[2]
          width = x.shape[3]

          '''
          the lower limits are defined as exclusive on the right,
          for this reason 1 is not subtracted
          '''
          max_min_top = height - self.size[0]
          max_min_left = width - self.size[1]
        
          if(max_min_top < 0 or max_min_left < 0):
             raise AssertionError('max_min_top < 0 or max_min_left < 0')
        
          if(max_min_top == 0): # here max_min_top > 0
             min_top = 0 
          else:
             min_top = np.random.randint(max_min_top) 

          if(max_min_left == 0): # here max_min_left > 0
             min_left = 0
          else:
             min_left = np.random.randint(max_min_left) 

          
          # note that pytorch slicing is exclusive on the right
          return x[:, :, min_top:min_top+self.size[0], min_left:min_left+self.size[1]]


class RandomFlipVideo():
      def __call__(self, x):
          a = np.random.random(1)[0]
          if(a >= 0.5):
              return torch.flip(x, [3])
          else:
              return x


class BagClipsDataset(Dataset):
      def __init__(self, path_positive_clips, path_negative_clips, length_window, stride, resize_size, crop_size, flag_flip = False):
          self.path_positive_clips = path_positive_clips
          self.path_negative_clips = path_negative_clips
          self.positive_clips = get_list_files(path_positive_clips, ['*.mp4', '*.avi'])
          self.negative_clips = get_list_files(path_negative_clips, ['*.mp4', '*.avi'])

          self.n_bags = len(self.positive_clips)
          self.length_window = length_window # this parameter must be in frames
          self.stride = stride # this parameter must be in frames

          # transformations
          # format of size: height = size[0], width = size[1]
          self.resize_size= resize_size
          self.crop_size= crop_size
          self.randomCropVideo = RandomCropVideo(self.crop_size) # height = size[0], width = size[1]
          self.randomFlipVideo = RandomFlipVideo()
          self.flag_flip = flag_flip
        
      def __len__(self):
          return self.n_bags 

      def __getitem__(self, idx):
          file_positive_clip = os.path.join(self.path_positive_clips, self.positive_clips[idx])
       
          idx_neg = random.randint(0, len(self.negative_clips) - 1)
          file_negative_clip = os.path.join(self.path_negative_clips, self.negative_clips[idx_neg])


          positive_bag = sample_clip(file_positive_clip, self.length_window, self.stride, self.resize_size)
          negative_bag = sample_clip(file_negative_clip, self.length_window, self.stride, self.resize_size)
       
          
          #######################TRANSFORMATIONS############################
          ##########CROP#############
          '''
          NOTE: the RandomCropVideo transformation is mandatory so that the samples of
          positive_bag and negative_bag have the same size and can be concatenated into
          the same tensor
          '''
          sz_posbag = positive_bag.shape[0]
          #height = size[0], width = size[1]
          positive_bag_cropped = torch.zeros(sz_posbag, 3, self.length_window, self.crop_size[0], self.crop_size[1], dtype=torch.uint8)
          for i, sample in enumerate(positive_bag):
              positive_bag_cropped[i] = self.randomCropVideo(sample)

          sz_negbag = negative_bag.shape[0]
          #height = size[0], width = size[1]
          negative_bag_cropped = torch.zeros(sz_negbag, 3, self.length_window, self.crop_size[0], self.crop_size[1], dtype=torch.uint8)
          for i, sample in enumerate(negative_bag):
              negative_bag_cropped[i] = self.randomCropVideo(sample)

          positive_bag = positive_bag_cropped
          negative_bag = negative_bag_cropped
          ###########################

          ##########FLIP#############
          if(self.flag_flip):
             for i, sample in enumerate(positive_bag):
                 positive_bag[i] = self.randomFlipVideo(sample)

             for i, sample in enumerate(negative_bag):
                 negative_bag[i] = self.randomFlipVideo(sample)
          ###########################

          ##################################################################


          data = torch.cat([positive_bag, negative_bag], axis=0)
          labels = torch.ones(data.shape[0], dtype=torch.float32)
          labels[negative_bag.shape[0]:] = -1.0*labels[negative_bag.shape[0]:]
          return data, labels


if __name__ == '__main__':

   '''
   number of samples is:
   n = 2*((fps*Ts - length_window)/stride) + 1
   where fps is frames per second of video and Ts is the time duration.
   For example:
   n = 2*(((8*3 - 8)/4) + 1)
   n = 10
   '''
   PATH_FILE = os.path.dirname(__file__)

   path_positive_clips = os.path.join(PATH_FILE,'data/positive')
   path_negative_clips = os.path.join(PATH_FILE,'data/negative')
   length_window = 8 #10, 12
   stride = 4 #5, 8
   bagClipsDataset = BagClipsDataset(path_positive_clips, path_negative_clips, length_window, stride, (185, 185), (150, 172), True)
   train_dataloader = DataLoader(bagClipsDataset, batch_size=1, shuffle=True)

   data = next(iter(train_dataloader))
   imshow_bags(data)
