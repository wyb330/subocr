import uuid
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import requests
import json
import os


BINARY_THREHOLD = 180


def temp_filename():
    return str(uuid.uuid4().hex)


def img2bytes(img, format='JPEG'):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes = img_bytes.getvalue()
    return img_bytes


def get_2points(box):
    '''
    사각형의 4점 좌표에서 2점 좌표(left, top, right, bottom)를 돌려준다.
    :param box: 사각형의 4점 좌표
    :return: 사각형의 2점 좌표(left, top, right, bottom)
    '''
    box = [(int(x), int(y)) for x, y in box]
    ps = np.reshape(box, [4, 2])
    p_min = ps.min(axis=0)
    p_max = ps.max(axis=0)
    left = p_min[0]
    top = p_min[1]
    right = p_max[0]
    bottom = p_max[1]

    return {'left': left, 'top': top, 'right': right, 'bottom': bottom}


def adjust_position(bboxes):
    boxes = []
    for i, box in enumerate(bboxes):
        box = get_2points(box)
        center = box['top'] + (box['bottom'] - box['top']) // 2
        boxes.append([i, box['left'], box['top'], box['right'], box['bottom'], center])

    for i in range(1, len(boxes)):
        prev_center = boxes[i - 1][5]
        center = boxes[i][5]
        if prev_center - 8 < center < prev_center + 8:
            boxes[i][5] = boxes[i - 1][5]

    arr = np.array(boxes)
    # height, left 순으로 정렬
    arr = list(arr[np.lexsort((arr[:, 1], arr[:, 5]))])
    ids = [v[0] for v in arr]
    adjust_boxes = [boxes[i] for i in ids]
    return adjust_boxes


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY_INV)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    # filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    # kernel = np.ones((1, 1), np.uint8)
    # opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    # or_image = cv2.bitwise_or(img, closing)
    return img


def convert_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    return img


def cv2pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def sub_image(img, rect):
    column, row, width, height = rect[0:4]
    img = img[row:row+height, column:column+width]
    return img


def detect_sub_area(image):
    """
    이미지에서 자막이 있는 영역을 찾는다.
    자막이 없는 영역은 흰색으로 채워져 있으므로 영역의 평균 및 표준편차 색상값으로 자막인지 여부를 판별한다.
    :param image: 자막이 있는 이미지
    :return:
    """
    (H, W) = image.shape[:2]
    index = 0
    d = 5
    threshold = 240

    while (index + 1) * d < H:
        img = sub_image(image, [0, index * d, W, d])
        avg = np.mean(img)
        std = np.std(img)
        if (avg < threshold) and (std > 10):
            break
        index += 1

    top = index * d
    return W, H, top


class TensorflowCallError(Exception):
    pass


def call_tf_serving(server_url, inputs):
    data = {"inputs": inputs}

    # Making POST request
    headers = {"content-type": "application/json"}
    json_response = requests.post(server_url, data=json.dumps(data), headers=headers)

    # Decoding results from TensorFlow Serving server
    json_text = json.loads(json_response.text)
    if 'outputs' in json_text:
        predictions = json_text['outputs']
    else:
        raise TensorflowCallError('TensorFlow Serving call failed. ' + json_response.text)
    return predictions


def save_box_images(filename, boxes, save_path):
    MARGIN = 2
    img = Image.open(filename)
    for i, box in enumerate(boxes):
        left, top, right, bottom, center = box[1:6]
        img_crop = img.crop([left - MARGIN, top - MARGIN, right + MARGIN, bottom + MARGIN])
        name, file_ext = os.path.splitext(os.path.basename(filename))
        path = os.path.join(save_path, '{}_{}{}'.format(name, i, file_ext))
        img_crop.save(path)


FONT_FILE = "./ocr/font/NanumBarunGothicBold.ttf"


def draw_image_text(filename, boxes, text):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = boxes[0][1:5]
    left_l, top_l, right_l, bottom_l = boxes[-1][1:5]
    font_size = bottom - top - 2
    font = ImageFont.truetype(FONT_FILE, font_size)
    lines = text.split('\n')
    long_text = lines[0]
    for line in lines:
        if len(line) > len(long_text):
            long_text = line
    max_width, _ = font.getsize(long_text)
    draw.rectangle((left - 2, top - 2, left + max_width + 2, bottom_l + 2), fill='white')
    draw.multiline_text((left, top), text, font=font, fill='black')

    img.save(filename)

