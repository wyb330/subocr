#**자막용 OCR**
이미지 파일들에 포함된 한글 자막/텍스트를 OCR를 이용하여 추출한 다음 자막파일 또는 텍스트 파일로
저장해주는 라이브러리입니다.

##**설치**
pip -r install requirements.txt

### tensorflow 설치

pip install tensorflow==1.11.0    # CPU를 사용하는 경우

또는 

pip install tensorflow-gpu==1.11.0   # GPU를 사용하는 경우

GPU를 사용하는 경우 CUDA 9.x 와 CUDNN이 먼저 설치되어 있어야 합니다. 

###학습 모델
학습된 모델을 아래 링크에서 다운로드 받아서 압축을 풉니다.
경로가 아래의 옵션에 설정한 기본값과 다르다면 실행시 경로를 지정해 주어야 합니다.

https://drive.google.com/file/d/1-mIAqb6hUSSxKkzClr7oCCfG9MsKuxCn/view?usp=sharing

##**사용법**
python subocr.py -i c:/tmp/sub -o ./sub.srt

###옵션
  -d : 텍스트 검출 모델의 경로. 기본값은 "./model/craft/weight.h5"
  
  -r : 텍스트 인식 모델 경로. 기본값은 "./model/aocr"
  
  -i : 이미지 경로
  
  -o : 출력 파일명. 기본값은 "sub.srt" 
  
##**학습**

[]: https://drive.google.com/file/d/1-mIAqb6hUSSxKkzClr7oCCfG9MsKuxCn/view?usp=sharing