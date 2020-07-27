'''
OCR  교정용 사전 구축
 - OCR 데이터에 대한 예측값과 살제값을 비교해 예측값이 잘못된 단어에 대해 사전을 구축한다.
'''
from utils import *
import tqdm
from argparse import ArgumentParser
import pandas as pd


def predict(model, image_file):
    img = Image.open(image_file)

    if image_file.endswith('.png'):
        img_format = 'PNG'
    else:
        img_format = 'JPEG'

    predictions = model({'input': [img2bytes(img, format=img_format)]})
    word = predictions['output'] if type(predictions['output']) is str else predictions['output'].decode('utf8')
    return word


def read_data(csv_file):
    data = pd.read_csv(csv_file, names=["filename", 'text'], sep=',', dtype='unicode', quotechar='"')
    images = list(data['filename'])
    labels = list(data['text'])

    return images, labels


def main(args):
    ocr_model = load_ocr_model(args.r)
    images, labels = read_data(args.i)
    images = images[:100000]
    labels = labels[:100000]
    d = {}
    with open(args.o, 'w', encoding='utf8') as f:
        for image, label in tqdm.tqdm(zip(images, labels), total=len(labels)):
            text = str(predict(ocr_model, image))
            label = str(label)
            if text.strip() != label.strip():
                d[text.strip()] = label.strip()
        json.dump(d, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", default="./model/aocr", help="텍스트 인식 모델 경로")
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", default="./correction.json", help="출력 파일명")
    args = parser.parse_args()
    main(args)
