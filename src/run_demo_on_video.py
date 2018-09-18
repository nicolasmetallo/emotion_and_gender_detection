import sys
import argparse

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""
    parser = argparse.ArgumentParser(description='Script to detect emotion and gender in videos')
    parser.add_argument('input_video', help='Path to input_video')
    parser.add_argument('path_to_extract', help='Path to extract frames')
    parser.add_argument('path_to_predict', help='Path to predicted frames')
    parser.add_argument('output_video', help='Path to output video')
    return parser

def run_emotion_gender_detector(input_video, path_to_extract, path_to_predict, output_video):
    try:
      os.mkdir(path_to_extract)
    except:
      pass

    # parameters for loading data and images
    image_path = path_to_extract
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # hyper-parameters for bounding boxes shape
    gender_offsets = (30, 60)
    gender_offsets = (10, 10)
    emotion_offsets = (20, 40)
    emotion_offsets = (0, 0)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    gender_classifier = load_model(gender_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]
    gender_target_size = gender_classifier.input_shape[1:3]

    print("Extracting frames from video...")
    os.system('ffmpeg -i {} -vf fps=30 {}/thumb%04d.png'.format(input_video, image_path)

    import glob
    import shutil
    from tqdm import trange

    images = glob.glob(os.path.join(image_path,"*.png"))
    images.sort()

    # Check if dir exists then remove, if not, then make dir
    if os.path.exists(path_to_predict):
      shutil.rmtree(path_to_predict)

    try:
      os.mkdir(path_to_predict)
    except:
      pass

    for i in trange(len(images)):
      # loading images
      rgb_image = load_image(images[i], grayscale=False)
      gray_image = load_image(images[i], grayscale=True)
      gray_image = np.squeeze(gray_image)
      gray_image = gray_image.astype('uint8')
      faces = detect_faces(face_detection, gray_image)
      for face_coordinates in faces:
          x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
          rgb_face = rgb_image[y1:y2, x1:x2]

          x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
          gray_face = gray_image[y1:y2, x1:x2]

          try:
              rgb_face = cv2.resize(rgb_face, (gender_target_size))
              gray_face = cv2.resize(gray_face, (emotion_target_size))
          except:
              continue

          rgb_face = preprocess_input(rgb_face, False)
          rgb_face = np.expand_dims(rgb_face, 0)
          gender_prediction = gender_classifier.predict(rgb_face)
          gender_label_arg = np.argmax(gender_prediction)
          gender_text = gender_labels[gender_label_arg]

          gray_face = preprocess_input(gray_face, True)
          gray_face = np.expand_dims(gray_face, 0)
          gray_face = np.expand_dims(gray_face, -1)
          emotion_prediction = emotion_classifier.predict(gray_face)
          emotion_probability = np.max(emotion_prediction)
          emotion_label_arg = np.argmax(emotion_prediction)
          emotion_text = emotion_labels[emotion_label_arg]

          if emotion_text == 'angry':
              color = emotion_probability * np.asarray((255, 0, 0))
          elif emotion_text == 'sad':
              color = emotion_probability * np.asarray((0, 0, 255))
          elif emotion_text == 'happy':
              color = emotion_probability * np.asarray((255, 255, 0))
          elif emotion_text == 'surprise':
              color = emotion_probability * np.asarray((0, 255, 255))
          else:
              color = emotion_probability * np.asarray((0, 255, 0))

          color = color.astype(int)
          color = color.tolist()

          draw_bounding_box(face_coordinates, rgb_image, color)
          draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
          draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

      bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
      cv2.imwrite('/content/face_classification/predicted_frames/predicted%04d.jpg' % i, bgr_image)

    print("Done! Merging frames into video...\n")
    # convert frames to video using FFMPEG
    os.system('ffmpeg -r 25 -f image2 -i {}/predicted%04d.jpg \
      -vcodec libx264 -crf 25 -pix_fmt yuv420p {}/output_video.mp4'.format(path_to_predict, path_to_predict))

if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    run_emotion_gender_detector(parsed_args.input_video, parsed_args.path_to_extract,
        parsed_args.path_to_predict, parsed_args.output_video)
