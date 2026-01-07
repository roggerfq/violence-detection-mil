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
import threading
import queue
import cv2
import numpy as np


class Sampler(threading.Thread):
      def __init__(self, path_video, new_fps = None, return_all_frames = False, size_batch=None, queue_max_size = None):
 
          self.cap = cv2.VideoCapture(path_video)
          if(not self.cap.isOpened()):
             if(isinstance(path_video, str)):
                self.cap.release()
                raise NameError('No se encuentra el archivo '+path_video)
             else:
                self.cap.release()
                raise NameError('No se encuentra el dispositivo') 
          #NOTE: improve error handling before releasing to CAP

          self.original_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
          if(self.original_fps <= 0):
             self.original_fps = 30

          if(new_fps is None):
             new_fps = int(self.original_fps)
          else:
              if(not isinstance(new_fps, int)):
                  raise NameError('new_fps must be an integer')
              elif(new_fps <= 0):
                  raise NameError('new_fps must be an integer greater than 0')

          self.new_fps = new_fps

          
          step = float(self.original_fps)/self.new_fps
          self.idxs = [int(float(i)*step) for i in range(0,self.new_fps)]

          if(queue_max_size is None):
             if(size_batch is None):
                queue_max_size = 2*self.original_fps #by default, we read 2 seconds ahead
                self.return_all_frames = return_all_frames
             else:
                queue_max_size = int(3 + float(2*self.original_fps)/size_batch)
                self.return_all_frames = False #in this case, we only return the true values

          self.size_batch = size_batch

          self.q = queue.Queue(maxsize = queue_max_size)
          threading.Thread.__init__(self)

          self.flag_started = False
          self.flag_stop = True

          self.flag_resize = False
          self.blank_image = np.zeros((int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                                       int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), 
                                       np.uint8)

      def resize(self, new_size):
          self.flag_resize = True
          if(isinstance(new_size, tuple)):
            self.new_width = new_size[0] 
            self.new_height = new_size[1]
          else:
            self.ratio = float(new_size)
            self.new_width = int(self.ratio * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.new_height = int(self.ratio * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

          self.blank_image = np.zeros((self.new_height, self.new_width, 3), np.uint8)

      def get_width(self):
          if(self.flag_resize):
             return self.new_width
          else:
             return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
      
      def get_height(self): 
          if(self.flag_resize):
             return self.new_height
          else:
             return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

      def get(prop):
          return self.cap.get(prop)

      def _get_frame(self):

          cnt = 0
          cnt_idx = 0

          while(not self.flag_stop):
                  ret, frame = self.cap.read()
                  if(ret):
                    if(cnt >= self.original_fps):
                       cnt = 0
                    i = (cnt % self.original_fps)
                    if(i == self.idxs[cnt_idx % self.new_fps]):          
                        if(self.flag_resize):
                           frame = cv2.resize(frame, (self.new_width, self.new_height), interpolation = cv2.INTER_AREA)
                        self.q.put((True, True, frame))
                        #cv2.imshow('img', frame)
                        #cv2.waitKey(1)
                        cnt_idx = cnt_idx + 1
                    else:
                        if(self.return_all_frames):
                          if(self.flag_resize):
                             frame = cv2.resize(frame, (self.new_width, self.new_height), interpolation = cv2.INTER_AREA)
                          self.q.put((True, False, frame))
                  else:
                        self.q.put((False, False, None))
                        break
                  cnt = cnt + 1


      def _get_batch(self):

          cnt = 0
          cnt_idx = 0
          batch_imgs = []
          while(not self.flag_stop):
                  ret, frame = self.cap.read()
                  if(ret):
                    if(cnt >= self.original_fps):
                       cnt = 0
                    i = (cnt % self.original_fps)
                    ###########################################
                    if(i == self.idxs[cnt_idx % self.new_fps]):          
                        if(self.flag_resize):
                           frame = cv2.resize(frame, (self.new_width, self.new_height), interpolation = cv2.INTER_AREA)

                        batch_imgs.append(frame)
                        if(len(batch_imgs) >= self.size_batch):
                           self.q.put((True, self.size_batch*[True], batch_imgs))
                           batch_imgs = []
                        cnt_idx = cnt_idx + 1
                    ###########################################
                  else:
                        if(len(batch_imgs) > 0):
                           flag_proc = [True for img in batch_imgs]
                           flag_proc = flag_proc + (self.size_batch - len(batch_imgs))*[False]
                           batch_imgs = batch_imgs + (self.size_batch - len(batch_imgs))*[self.blank_image]
                           self.q.put((True, flag_proc, batch_imgs))
                        
                        flag_proc = self.size_batch*[False]
                        batch_imgs = self.size_batch*[self.blank_image]
                        self.q.put((False, flag_proc, batch_imgs))
                           
                        break

                  cnt = cnt + 1
           
      def run(self):
          if(self.size_batch is not None):
            self._get_batch()
          else:
            self._get_frame()
             

      def read(self):
          if(not self.flag_started):
             self.flag_started = True
             self.flag_stop = False
             self.start()
                
          data = self.q.get()
          ret = data[0]
          proc_flag = data[1]
          data_frame = data[2]
          if(not ret):
             self.read = self.read_end
          return ret, proc_flag, data_frame

      def read_end(self):
          return False, self.size_batch*[False], self.size_batch*[self.blank_image]

      def release(self):
          while(not self.q.empty()):
               self.q.get()
          if(not self.flag_stop):
             self.flag_stop = True
             self.join()
          self.cap.release()
          self.read = self.read_end



if __name__ == "__main__":

   '''
   The following code tests some of the 
   functionalities implemented in this file
   '''
  
   cap = Sampler('/path_to_video/video_test.mp4', 10, True)

   while(True):
        ret, flag_proc, frame = cap.read()
        print(flag_proc)
        if(ret):
           if(flag_proc):
             cv2.imshow('img', frame)
             if cv2.waitKey(1) & 0xFF == ord('q'):
                break
           else:
             pass
        else:
            break

   ret, flag_proc, frame = cap.read()
   cap.release()

