import tensorflow as tf
import mediapipe as mp
import cv2
from deepface import DeepFace

print("TensorFlow:", tf.__version__)
print("MediaPipe:", mp.__version__)
print("OpenCV:", cv2.__version__)
DeepFace.stream(db_path=None)
