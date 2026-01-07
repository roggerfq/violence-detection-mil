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


import cv2
import threading
import queue
import time

def sample_frames(path_video, q_frames, target_fps=8):
    """
    Extract frames from video and put them in queue at target FPS.

    Args:
        path_video: Path to the video file
        q_frames: Queue to store frames
        target_fps: Target frames per second (default 8)
    """
    cap = cv2.VideoCapture(path_video)

    if not cap.isOpened():
        print(f"Error: Cannot open video {path_video}")
        q_frames.put(None)  # Signal end
        return

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate expected output frames
    duration = total_frames / original_fps
    expected_frames = int(duration * target_fps)

    print(f"Video FPS: {original_fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
    print(f"Expected output frames at {target_fps} fps: {expected_frames}")

    frame_count = 0
    extracted_count = 0
    next_frame_time = 0  # This is the time (in frame indices) when we must extract next frame

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Check if we should extract this frame based on target fps timing
        if frame_count >= next_frame_time:
            q_frames.put(frame.copy())
            extracted_count += 1
            # Calculate next extraction time
            next_frame_time = (extracted_count * original_fps) / target_fps

        frame_count += 1

    cap.release()
    print(f"Extraction complete: {extracted_count} frames extracted from {frame_count} total frames")

    # Signal end of video
    q_frames.put(None)

def process_frames(model, q_frames, length_window, stride, scores):
    """
    Process frames from queue using model and store results in scores list.

    Args:
        model: Model object for evaluation (should have a method to process frames)
        q_frames: Queue containing frames to process
        scores: List to store processing results
    """
    processed_count = 0


  
    clip = []
    while True:
        try:
            frame = q_frames.get(timeout=5)  # 5 second timeout

            # None signals end of video
            if frame is None:
                print(f"Processing complete: {processed_count} frames processed")
                break

            # Processing clip 
            clip.append(frame)
            if(len(clip) >= length_window):
               result = model([clip])
               scores.append(result.item())
               clip = clip[stride:]



            processed_count += 1

            q_frames.task_done()

        except queue.Empty:
            print("Queue timeout - no frames received")
            break
        except Exception as e:
            print(f"Error processing frame {processed_count}: {e}")
            continue


def run_video_processing(path_video, model, target_fps, length_window, stride, max_queue_size=50):
    """
    Main function to run video processing with threading.

    Args:
        path_video: Path to video file
        model: Model for frame evaluation
        target_fps: Target sampling rate (default 8 fps)
        max_queue_size: Maximum queue size to prevent memory overflow

    Returns:
        scores: List of processing results
    """
    # Create queue and scores list
    q_frames = queue.Queue(maxsize=max_queue_size)
    scores = []

    # Create threads
    thread_sample = threading.Thread(
        target=sample_frames,
        args=(path_video, q_frames, target_fps),
        name="FrameSampler"
    )

    thread_processing = threading.Thread(
        target=process_frames,
        args=(model, q_frames, length_window, stride, scores),
        name="FrameProcessor"
    )

    # Start threads
    print("Starting video processing...")
    start_time = time.time()

    thread_sample.start()
    thread_processing.start()

    # Wait for both threads to finish
    thread_sample.join()
    thread_processing.join()

    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    print(f"Total frames processed: {len(scores)}")

    return scores