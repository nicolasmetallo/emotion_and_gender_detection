# Implementation of emotion and gender detection using TensorFlow
Please take a look at the [original repository](https://github.com/oarriaga/face_classification) and to their [publication](https://github.com/oarriaga/face_classification/blob/master/report.pdf)

# Real-time demo:
Real-time demo:
<div align='center'>
  <img src='images/color_demo.gif' width='400px'>
</div>

## Instructions for Google Colab

### Clone this repo:
```
$ git clone https://github.com/nicolasmetallo/emotion_and_gender_detection.git
```

### Pre-requisites:
Run the following to install the required modules
```
$ pip install -r face_classification/REQUIREMENTS.txt
```

### Install FFMPEG for Google Colab:
```
$ apt install ffmpeg
```

### Set root dir
```
$ cd face_classification/src
```

### Example usage
```
$ python run_demo_on_video.py emotion-gender-demo.mp4 \
      /content/emotion_and_gender_detection/extracted_frames \
      /content/emotion_and_gender_detection/predicted_frames \
      output_video.mp4
```
