# VD-MIL: A Deep Multiple Instance Learning Approach for Violence Detection in Surveillance Videos

VD-MIL is an algorithm for violence detection in surveillance videos inspired by the Multiple Instance Learning approach presented by Sultani at CVPR 2019 [1]. The original algorithm used a very long convolutional network known as C3D for feature extraction and used the representation to train a binary classifier using MIL methodology with the UCF-crime dataset. While this work presents remarkable results, it is not optimized to run on devices with hardware constraints without a GPU. This implementation replaces C3D with MoViNet [2], which is lighter, presents robust representation, and has been optimized to run on CPU. Additionally, this work trains the model to focus exclusively on fights and shooting by combining the training sets of the RWF-2000 [3] and Smart-City CCTV Violence Detection Dataset (SCVD) [4]. The implementation is kept simple using only PyTorch as a deep learning library.

The following link hosts a live demo of the algorithm:

[![Hugging Face Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow)](https://roggerfq-violence-detection.hf.space/)


<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/roggerfq/violence-detection-mil/refs/heads/main/docs/VD-MIL.gif" width="100%">
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/roggerfq/violence-detection-mil/refs/heads/main/docs/VD-MIL-2.gif" width="100%">
    </td>
  </tr>
</table>

## Installation
First, install the following fork of the MoViNet library:
```bash
pip install git+https://github.com/roggerfq/MoViNet-pytorch.git
```

Unlike the original repository, this fork allows selection of the download path for MoViNet pretrained weights.

Lastly, install this repository:
```bash
git clone https://github.com/roggerfq/violence-detection-mil.git
```

## Testing
To quickly run the algorithm with an example video, open and execute the following Google Colab notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E0x7aUeDGNI6gDqx4jPbNctQ0hU6kKY1?usp=sharing)

For local execution, run the following Python script:
```python
from movinet_classifier import MovinetClassifier
from activity_detection import run_video_processing
from signal_smoothing import smooth_scores
import matplotlib.pyplot as plt

path_movinet_model = './movinet_weights' #Backbone model 
detector_models = ['./test/model_20.pt'] #Classifier model
classifier = MovinetClassifier(path_movinet_model, detector_models, device = 'cpu')

path_video = './test/video_test.mp4'
target_fps = 8 #8fps
length_window =8 #8 frames
stride = 4 #4 frames

scores = run_video_processing(path_video, classifier, target_fps, length_window, stride)
scores_smooth = smooth_scores(scores)#Savitzky-Golay filter

# Each stride produces a score; to convert it back to seconds, we divide by target_fps
t_scores = [(i*stride)/target_fps for i in range(len(scores_smooth))]


plt.figure()
plt.plot(t_scores, scores_smooth)
plt.xlabel("Time (s)")
plt.ylabel("Score")
plt.title("Scores Over Time")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
```

## Training
The training requires two major steps. First, generate positive and negative video clip segments from the raw dataset. Then, using those video clip segments, train the architecture comprised of a MoViNet backbone and a fully-connected binary classifier. The following image illustrates this process:

![Full Architecture](https://raw.githubusercontent.com/roggerfq/violence-detection-mil/main/docs/full_architecture.svg)

### Video clip generation

For optimization purposes, both at training and test time, the algorithm processes videos at a lower `fps` rate than the original. Therefore, before training, the video set must be downsampled to a lower and fixed `fps`. The following example shows how to create a new set of videos with a fixed `fps` of 8:

```bash
python normalize_fps.py 8 /path_to_original_videos /videos_8fps
```

After downsampling the videos, the next step is to extract small video clip segments from each video. To perform this step, the folder containing the downsampled videos must also contain a `.txt` file with the ground truth information. The following example shows the structure this file must have:

```txt
video_1.mp4 10,34,45,56
video_2.mp4 0,inf
video_3.mp4 -1,-1
.
.
.
video_n.mp4 5,20
```

In the previous example, the first column contains the video name and the second column contains time pairs indicating the beginning and ending of the target event (violence in this repository). For example, `video_1.mp4` contains violence events between seconds 10 and 34, and between seconds 45 and 56. If ground truth information is unavailable but the video is known to contain the target event, the ground truth must be set to `0,inf`, indicating the entire video is a positive event. Conversely, if the video is known not to contain the target event, the ground truth must be set to `-1,1`. With the downsampled video set and ground truth information, the script `generate_clips_dataset.py` applies a temporal window of length `lenght_clip` seconds that moves across each video with a fixed stride of `stride_window_clip` seconds. At each position, it computes the relative overlap between this window and the ground truth; if this value exceeds a threshold (0.3 by default), the segment is stored in the positive clip set. A segment is stored in the negative clip set only if the overlap is zero. The following example shows how to generate the video clip segments:

```bash
python generate_clips_dataset.py \
  --path_set_videos /videos_8fps \
  --path_clips /clips/Violence \
  --label_file /videos_8fps/labels.txt \
  --new_fps 8 \
  --length_clip 5 \
  --stride_window_clip 5 \
  --overlap_positive_clip 0.3
```

To quickly run this process, open and execute the following Google Colab notebooks, which generate video clip segments using the training sets of the RWF-2000 and SCVD datasets respectively.

**Video clip generation using the RWF-2000 dataset**: 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12Ta7GBYICHf0tkQPtc2SmEIKVwkMa3xl?usp=sharing)

**Video clip generation using the SCVD dataset**:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MbA0jKGNbAY2zpz-m6_Bq470R5eOub8I?usp=sharing)

### Training the model

The previous step generates two folders containing the positive and negative video clip segments respectively. In this step, these segments are used to train the model, which comprises a MoViNet feature extractor and a fully-connected network as a binary classifier. In each forward step, the algorithm randomly selects a set of positive and negative clips to create the positive bag and negative bag respectively. The video clips are not fed into MoViNet entirely; instead, they are first segmented using a temporal window of length `length_window` frames that moves across the clip with a fixed stride of `stride` frames. The following example shows how to run the training:

```bash
python train_activity_detector.py \
  --path_positive_clips /clips/Violence/positive \
  --path_negative_clips /clips/Violence/negative \
  --folder_backbone_model ./movinet_weights \
  --batch_size 16 \
  --epochs 30 \
  --checkpoint_interval 1 \
  --length_window 8 \
  --stride 4 \
  --folder_trained_models /results/training/ \
  --device cuda
```

The following Google Colab notebook uses pre-generated video segments shared via a public Google Drive link to train the model.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GH8xWBtTnY0aV-FmT9jWk_tSmGgmQk3u?usp=sharing)

## Evaluation

The algorithm evaluation uses the evaluation set of the RWF-2000 dataset. This set contains 200 short video segments containing scenes of violence and 200 short video segments that do not contain any type of violence. The following figure shows the ROC curve and AUC score obtained in this evaluation.

![ROC curve](https://raw.githubusercontent.com/roggerfq/violence-detection-mil/refs/heads/main/results/roc_curve.png)

## Author
Roger Figueroa Quintero - [LinkedIn Profile](https://www.linkedin.com/in/roger-figueroa-quintero/)

## License
This project is licensed under the [MIT License](LICENSE.md), allowing unrestricted use, modification, and distribution under the terms of the license.

## References

[1] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6479–6488.

[2] D. Kondratyuk, L. Yuan, Y. Li, L. Zhang, M. Tan, M. Brown, and B. Gong, “MoViNets: Mobile video networks for efficient video recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 16020–16030.

[3] M. Cheng, K. Cai, and M. Li, “RWF-2000: An open large scale video database for violence detection,” in 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 4183–4190.

[4] T. Aremu, Z. Li, R. Alameeri, M. Khan, and A. El Saddik, “SSIVD-Net: A novel salient super image classification and detection technique for weaponized violence,” in Intelligent Computing, K. Arai, Ed. Cham, Switzerland: Springer Nature Switzerland, 2024, pp. 16–35.
