from zipfile import ZipFile


with ZipFile("archive (1).zip","r")as zip:
    zip.extractall()

import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm

# Load ResNet50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Feature extraction function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0).flatten()
    normalized_features = features / norm(features)

    return normalized_features

# Get image filenames
image_dir = "images"
filenames = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# Extract features
feature_list = []
for img_path in tqdm(filenames):
    feature_list.append(extract_features(img_path, model))

# Save features
with open("feature_list.pkl", "wb") as f:
    pickle.dump(feature_list, f)

with open("filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)
    
