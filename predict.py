# import tensor flow librarys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

#import some important libraries
import sys
import time 
import numpy as np
import matplotlib.pyplot as plt

# For runuing the script
from PIL import Image
import argparse
import json

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Some costants
batch_size = 32


# Transform the image to the right format
def process_img(img): 
    img_size = 224
    img = tf.cast(img, tf.float32)
    img = tf.img.resize(img, (img_size, img_size))
    img /= 255    
    return img.numpy()


# presdiction function
def predict_class(img_path, model, top_k=5):
    processed_image = process_img(np.asarray(Image.open(img_path)))
    img_predicted = np.expand_dims(processed_image, axis=0)
    preds = model.predict(img_predicted)
    probs = - np.partition(-preds[0], top_k)[:top_k]
    classes = np.argpartition(-preds[0], top_k)[:top_k]  

    return probs, classes


# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('img_path', action = "store")
    parser.add_argument('saved_model', action = "store")
    parser.add_argument('--top_k', action = "store", dest = "top_k", type = int, default=5)
    parser.add_argument('--category_names', action = "store", dest = "category_names")

    
    results = parser.parse_args()
    print(results)
    
    top_k = results.top_k
    img_path = results.img_path
    saved_model = results.saved_model
    category_filename = results.category_names
    print('image_path:', img_path)
    print('saved_model:', saved_model)
    print('top_k:', top_k)
    print('category_names:', category_filename)

    model = tf.keras.models.load_model(saved_model ,custom_objects={'KerasLayer':hub.KerasLayer} )
    
    image = np.asarray(Image.open(img_path)).squeeze()
    probs, classes = predict_class(img_path, model, top_k)

    with open(category_filename, 'r') as f:
        class_names = json.load(f)
    keys = [str(x+1) for x in list(classes)]
    classes = [class_names.get(key) for key in keys]

    print('Top {} classes are:'.format(top_k))
    for i in np.arange(top_k):
        print('Class: {}'.format(classes[i]))
        print('Probability: {:.3%}'.format(probs[i]))
        print('===================================\n')