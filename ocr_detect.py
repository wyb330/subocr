# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: predict.py

import cv2
import argparse
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from detection.net.vgg16 import VGG16_UNet
from detection.utils.inference_util import getDetBoxes, adjustResultCoordinates
from detection.utils.img_util import load_image, img_resize, img_normalize, to_heat_map


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='./model/craft/weight.h5', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1920, type=int, help='image size for inference')
parser.add_argument('--image_file', default=r'd:/data/subtitle/kr/test/real_images/text/screen1.png',
                    type=str, help='input image')

FLAGS = parser.parse_args()


def predict(model, image, text_threshold, link_threshold, low_text):
    h, w = image.shape[:2]
    if (h < 10) or (w < 10):
        return [], []
    mag_ratio = 1
    img_resized, target_ratio = img_resize(image, mag_ratio, min(FLAGS.canvas_size, w), interpolation=cv2.INTER_LINEAR)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = img_normalize(img_resized)

    # make score and link map
    score_text, score_link = model.predict(np.array([x]))
    score_text = score_text[0]
    score_link = score_link[0]

    # Post-processing
    boxes = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    # render results (optional)
    render_img = score_text.copy()
    white_img = np.ones((render_img.shape[0], 10, 3), dtype=np.uint8) * 255
    ret_score_text = np.hstack((to_heat_map(render_img), white_img, to_heat_map(score_link)))

    return boxes, ret_score_text


def detect(model, image_file):
    image = load_image(image_file)
    bboxes, score_text = predict(model, image, FLAGS.text_threshold, FLAGS.link_threshold, FLAGS.low_text)
    return bboxes


def detect_img(model, image):
    bboxes, score_text = predict(model, image, FLAGS.text_threshold, FLAGS.link_threshold, FLAGS.low_text)
    return bboxes


def load_detect_model(model_path):
    print('loading saved ocr detection model from - {}'.format(model_path))
    input_image = Input(shape=(None, None, 3), name='image', dtype=tf.float32)
    region, affinity = VGG16_UNet(input_tensor=input_image, weights=None)
    model = Model(inputs=[input_image], outputs=[region, affinity])
    model.load_weights(model_path)
    model._make_predict_function()

    return model


def main():
    """ Load model """
    input_image = Input(shape=(None, None, 3), name='image', dtype=tf.float32)
    region, affinity = VGG16_UNet(input_tensor=input_image, weights=None)
    model = Model(inputs=[input_image], outputs=[region, affinity])
    model.load_weights(FLAGS.trained_model)

    image_file = FLAGS.image_file
    image = load_image(image_file)
    bboxes, score_text = predict(model, image, FLAGS.text_threshold, FLAGS.link_threshold, FLAGS.low_text)
    return bboxes


if __name__ == '__main__':
    main()
