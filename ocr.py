from utils import *
from argparse import ArgumentParser


MARGIN = 4
FONT_FILE = "gulim.ttc"
FONT_SIZE = 14
LINE_HEIGHT = 10


def predict_image(model, detect_model, image_file, result_path=None):
    '''
    :param model: 텍스트 인식 모델
    :param detect_model: 텍스트 검출 모델
    :param image_file: 이미지 파일
    :param result_path: OCR 인식결과 이미지 파일
    :return: OCR에서 인식한 텍스트
    '''

    def extract_words(img_list, eol_list):
        words = []
        probs = []
        predictions = model({'input': img_list})
        if (type(predictions['output']) is bytes) or (type(predictions['output']) is str):
            word = predictions['output'] if type(predictions['output']) is str else predictions['output'].decode('utf8')
            words.append(word)
            probs.append(predictions['probability'])
        else:
            for output, prob, eol in zip(predictions['output'],  predictions['probability'], eol_list):
                word = output if type(output) is str else output.decode('utf8')
                probs.append(prob)
                if eol:
                    words.append('\n' + word)
                else:
                    words.append(word)

        return words, probs

    def draw_box_word(words, probs, boxes):
        for word, probability, box in zip(words, probs, boxes):
            left, top, right, bottom, center = box[1:6]
            word_width, word_height = font.getsize(word)
            prob_width, _ = font.getsize('{:.2f}'.format(probability))
            if img_height >= min_height:
                point_border = [(left + MARGIN, top - LINE_HEIGHT - FONT_SIZE * 2),
                                (left + max(word_width, prob_width) + MARGIN * 2,
                                 top - LINE_HEIGHT - FONT_SIZE + word_height)]
                point_text = (left + MARGIN * 2, top - LINE_HEIGHT - FONT_SIZE * 2)
                point_prob = (left + MARGIN * 2, top - LINE_HEIGHT - FONT_SIZE)
            else:
                point_border = [(left + MARGIN, top + word_height + LINE_HEIGHT + FONT_SIZE * 3),
                                (left + max(word_width, prob_width) + MARGIN * 2,
                                 top + word_height + LINE_HEIGHT + FONT_SIZE * 4 + word_height)]
                point_text = (left + MARGIN * 2, top + word_height + LINE_HEIGHT + FONT_SIZE * 3)
                point_prob = (left + MARGIN * 2, top + word_height + LINE_HEIGHT + FONT_SIZE * 4)

            draw.rectangle([(max(1, left - MARGIN), max(1, top - MARGIN)),
                            (min(img_width - 1, right + MARGIN), min(img_height - 1, bottom + MARGIN))],
                           outline='green',
                           width=2)
            draw.rectangle(point_border, fill="black")
            draw.text(point_text, word.strip(), font=font, fill="white")
            draw.text(point_prob, '{}%'.format(int(probability * 100)), font=font, fill="white")

    min_height = 200
    try:
        img = Image.open(image_file)
    except OSError:
        raise ValueError('이미지 파일을 읽는데 실패했습니다.')

    boxes = run_detect(detect_model, image_file)
    if len(boxes) == 0:
        print('텍스트 검출 실패: {}'.format(os.path.basename(image_file)))
        return ''

    if image_file.lower().endswith('.png'):
        img_format = 'PNG'
    else:
        img_format = 'JPEG'

    eol_list = []
    img_list = []
    prev_center = 0
    boxes = adjust_position(boxes)
    for i, box in enumerate(boxes):
        left, top, right, bottom, center = box[1:6]
        img_crop = img.crop([left - MARGIN, top - MARGIN, right + MARGIN, bottom + MARGIN])
        img_list.append(img2bytes(img_crop, format=img_format))

        if (prev_center > 0) and (center != prev_center):
            eol_list.append(True)
        else:
            eol_list.append(False)
        prev_center = center

    words, probs = extract_words(img_list, eol_list)

    # OCR 인식 결과(텍스트 영역,텍스트,확률)를 이미지로 저장한다.
    if result_path is not None:
        font = ImageFont.truetype(FONT_FILE, FONT_SIZE)
        img_width = img.width
        img_height = img.height
        if img_height < min_height:
            new_img = Image.new('RGB', (img_width, img_height + FONT_SIZE * 5), (255, 255, 255, 255))
            new_img.paste(img, (0, 0))
            img = new_img
        draw = ImageDraw.Draw(img)
        draw_box_word(words, probs, boxes)
        img.save(result_path, format=img_format)

    text = ' '.join(words)
    return text


def ocr(model, detect_model, image_file, result_path=None):
    text = predict_image(model, detect_model, image_file, result_path)
    return text


def main(args):
    if not allowed_image_file(args.i):
        print('이미지 파일이 아니거나 지원하지 않는 이미지 형식입니다.')
        return
    ocr_model = load_ocr_model(args.r)
    detect_model = load_detect_model(args.d)
    text = ocr(ocr_model, detect_model, args.i, result_path=args.o)
    # OCR 인식 결과를 교정한다.
    text = ocr_correction(text, 'correction.json')
    print(text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", default="./model/craft/weight.h5", help="텍스트 검출 모델 경로")
    parser.add_argument("-r", default="./model/aocr", help="텍스트 인식 모델 경로")
    parser.add_argument("-i", required=True, help="이미지 파일")
    parser.add_argument("-o", help="OCR 인식결과 이미지 파일명")
    args = parser.parse_args()
    main(args)
