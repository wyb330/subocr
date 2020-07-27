# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: predict.py

import cv2
import numpy as np
from detection.utils.inference_util import getDetBoxes, adjustResultCoordinates
from detection.utils.img_util import load_image, img_resize, img_normalize, to_heat_map
from collections import namedtuple


CraftFlags = namedtuple('CraftFlags', 'trained_model text_threshold low_text link_threshold canvas_size image_file')
FLAGS = CraftFlags(trained_model='./model/craft/weight.h5',
                   text_threshold=0.7,
                   low_text=0.4,
                   link_threshold=0.4,
                   canvas_size=5000,  # 이미지의 최대 크기
                   image_file='')


def predict(model, image):
    text_threshold = FLAGS.text_threshold
    link_threshold = FLAGS.link_threshold
    low_text = FLAGS.low_text
    h, w = image.shape[:2]
    if (h < 10) or (w < 10):
        return [], []
    mag_ratio = 1
    img_resized, target_ratio = img_resize(image, mag_ratio, min(FLAGS.canvas_size, max(w, h)), interpolation=cv2.INTER_LINEAR)
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
    bboxes, score_text = predict(model, image)
    return bboxes


def detect_img(model, image):
    bboxes, score_text = predict(model, image)
    return bboxes

