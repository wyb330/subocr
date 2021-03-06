import glob
import re
from bs4 import BeautifulSoup
from utils import *
from ocr import predict_image
import tqdm
from argparse import ArgumentParser


MARGIN = 8
VOBTIMECODE1 = '(#\d+:)(\d+:*\d+,\d+->\d+:\d+,\d+)(.)*'
VOBTIMECODE2 = '(#\d+:)(\d+:\d+:*\d+,\d+->\d+:\d+:\d+,\d+)(.)*'
VOBTIMECODE3 = '(#\d+:)(\d+,\d+->\d+,\d+)(.)*'


def predict_file(model, detect_model, image_file, auto_crop=True):
    text = predict_image(model, detect_model, image_file)
    return text


# videosubfinder 자막 이미지 파일명 형식인지 체크
def is_timecode_filename(filename):
    if re.match('\d+_\d{2}_\d{2}_\d{1,3}__\d+_\d{2}_\d{2}_\d{1,3}', filename):
        return True
    else:
        return False


def to_srt_timestamp(total_seconds):
    total_seconds = total_seconds / 1000
    hours = int(total_seconds / 3600)
    minutes = int(total_seconds / 60 - hours * 60)
    seconds = int(total_seconds - hours * 3600 - minutes * 60)
    milliseconds = round((total_seconds - seconds - hours * 3600 - minutes * 60)*1000)

    return '{:02d}:{:02d}:{:02d}.{:03d}'.format(hours, minutes, seconds, milliseconds)


def str2time(time):
    t = time.split('_')
    seconds = int(t[0]) * 3600 + 60 * int(t[1]) + int(t[2]) + (int(t[3]) / 1000)
    return int(seconds * 1000)


def filename2timecode(file):
    sub_t = '.'.join(file.split('.')[:-1])
    if sub_t.endswith('!'):
        sub_t = str(sub_t[:-1])
    sub_t = sub_t.split('__')
    if len(sub_t) == 1:
        return file
    start = to_srt_timestamp(str2time(sub_t[0]))
    end = to_srt_timestamp(str2time(sub_t[1]))
    return '{} --> {}'.format(start, end)


def ocr_subtitle(model, detect_model, path, files, timecodes, auto_crop):
    line_id = 1
    lines = []
    for file, t in tqdm.tqdm(zip(files, timecodes), total=len(files)):
        img_file = os.path.join(path, file)
        if not os.path.exists(img_file):
            continue
        if not allowed_image_file(img_file):
            continue
        text = predict_file(model, detect_model, img_file, auto_crop)
        if t is not None:
            lines.append('{}\n'.format(line_id))
            lines.append('{}\n'.format(t))
            lines.append('{}\n'.format(text))
            lines.append('\n')
        else:
            lines.append(text)
            lines.append('\n')

        line_id += 1

    text = ''.join(lines)
    return text


def vob2timecode(vob):

    def s2t(time):
        ts = time.split(',')
        ms = ts[1]
        t = ts[0].split(':')
        if len(t) == 3:
            seconds = int(t[0]) * 3600 + 60 * int(t[1]) + int(t[2]) + (int(ms) / 1000)
        elif len(t) == 2:
            seconds = 60 * int(t[0]) + int(t[1]) + (int(ms) / 1000)
        else:
            seconds = int(t[0]) + (int(ms) / 1000)
        return int(seconds * 1000)

    sub_t = vob.split('->')
    start = to_srt_timestamp(s2t(sub_t[0]))
    end = to_srt_timestamp(s2t(sub_t[1]))
    return '{} --> {}'.format(start, end)


def sub_timecode(regex1, regex2, regex3, line):
    m = regex1.match(line)
    if m is not None:
        timecode = m.group(2)
        timecode = vob2timecode(timecode)
    else:
        m = regex2.match(line)
        if m is not None:
            timecode = m.group(2)
            timecode = vob2timecode(timecode)
        else:
            m = regex3.match(line)
            if m is not None:
                timecode = m.group(2)
                timecode = vob2timecode(timecode)
            else:
                timecode = line
    return timecode


def vobsub_timecodes(html_file):
    with open(html_file, 'r', encoding='utf8') as f:
        soup = BeautifulSoup(f, "html.parser")
        lines = soup.get_text().split('\n')
    regex1 = re.compile(VOBTIMECODE1)
    regex2 = re.compile(VOBTIMECODE2)
    regex3 = re.compile(VOBTIMECODE3)
    lines = [sub_timecode(regex1, regex2, regex3, line) for line in lines if len(line) > 0 and line[0] == '#']
    return lines


def ocr(model, detect_model, path, auto_crop=True):
    files = glob.glob(os.path.join(path, '*.*'))
    if len(files) == 0:
        return '', 0

    filecount = len(files)
    try:
        # Videosubfinder에서 추출한 자막 이미지인 경우
        if is_timecode_filename(os.path.basename(files[-1])):
            timecodes = [filename2timecode(os.path.basename(file)) for file in files]
        # Subtitle Edit에서 추출한 VobSub 이미지인 경우
        elif os.path.basename(files[-1]) == 'index.html':
            timecodes = vobsub_timecodes(os.path.join(path, files[-1]))
            filecount -= 1
        else:
            timecodes = [None for _ in files]

        if filecount != len(timecodes):
            raise ValueError("이미지의 갯수({})와 타임코드의 갯수({})가 일치하지 않습니다.".format(filecount, len(timecodes)))
        text = ocr_subtitle(model, detect_model, path, files, timecodes, auto_crop)
    finally:
        pass
    return text, len(files)


def main(args):
    ocr_model = load_ocr_model(args.r)
    detect_model = load_detect_model(args.d)
    text, count = ocr(ocr_model, detect_model, args.i, auto_crop=False)
    # OCR 인식 결과를 교정한다.
    text = ocr_correction(text, 'correction.json')
    if count == 0:
        print('파일이 존재하지 않습니다.')
    else:
        with open(args.o, 'w', encoding='utf8') as f:
            f.write(text)

        print('OCR 인식 결과를 {} 파일로 저장했습니다.'.format(args.o))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", default="./model/craft/weight.h5", help="텍스트 검출 모델 경로")
    parser.add_argument("-r", default="./model/aocr", help="텍스트 인식 모델 경로")
    parser.add_argument("-i", required=True, help="이미지 경로")
    parser.add_argument("-o", default="./sub.srt", help="출력 파일명")
    args = parser.parse_args()
    main(args)
