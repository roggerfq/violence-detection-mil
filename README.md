# VD-MIL: A Deep Multiple Instance Learning Approach for Violence Detection in Surveillance Videos

VD-MIL is an algorithm for violence detection in surveillance videos inspired by the Multiple Instance Learning approach presented by Sultani at CVPR 2019. The original algorithm used a very long convolutional network known as C3D for feature extraction and used the representation to train a binary classifier using MIL methodology with the UCF-crime dataset. While this work presents remarkable results, it is not optimized to run on devices with hardware constraints without a GPU. This implementation replaces C3D with MoViNet, which is lighter, presents robust representation, and has been optimized to run on CPU. Additionally, this work trains the model to focus exclusively on fights, disturbances, and shooting by combining the training sets of the RWF-2000 and Smart-City CCTV Violence Detection Dataset (SCVD). The implementation is kept simple using only PyTorch as a deep learning library.

**Demo**: The following **Hugging Face** link hosts a demo of the algorithm → [**Link**](https://roggerfq-violence-detection.hf.space/)

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
from movinet_classifier import MovinetClassifier, Net #Net must be imported
from activity_detection import run_video_processing
from signal_smoothing import smooth_scores
import matplotlib.pyplot as plt

path_movinet_model = './movinet_weights'
detector_models = ['./results/model.pt']
path_video = './test/test.mp4'

classifier = MovinetClassifier(path_movinet_model, detector_models, device = 'cpu')
scores = run_video_processing(path_video, classifier, 8, 8, 4)
scores_smooth = smooth_scores(scores)

plt.figure()
plt.plot(scores_smooth)
plt.xlabel("Index")
plt.ylabel("Score")
plt.title("Scores Over Time")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
```

## Training
The training requires two major steps. First, generate positive and negative video clip segments from the raw dataset. Then, using those video clip segments, train the architecture comprised of a MoViNet backbone and a 3-layer fully-connected binary classifier. The following image illustrates this process:

![Full Architecture](https://raw.githubusercontent.com/roggerfq/violence-detection-mil/main/docs/full_architecture.svg)

To run this process, open and execute the following three Google Colab notebooks in order. The first and second notebooks contain the process for video clip segment generation using the training set of the RWF-2000 and SCVD datasets respectively. These two notebooks can be skipped; proceed directly to the training notebook, which uses pre-generated video segments shared via a public Google Drive link.

**Video clip generation using the RWF-2000 dataset**: 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12Ta7GBYICHf0tkQPtc2SmEIKVwkMa3xl?usp=sharing)

**Video clip generation using the SCVD dataset**:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MbA0jKGNbAY2zpz-m6_Bq470R5eOub8I?usp=sharing)

**Training**:

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

[2] D. Kondratyuk, L. Yuan, Y. Li, L. Zhang, M. Brown, and B. Gong, "MoViNets: Mobile Video Networks for Efficient Video Recognition," arXiv preprint arXiv:2103.11511, 2021.
