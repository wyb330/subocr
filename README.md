# **자막용 OCR**
이미지 파일들에 포함된 한글 자막/텍스트를 OCR를 이용하여 추출한 다음 자막파일 또는 텍스트 파일로
저장해주는 라이브러리입니다.

## **설치**
1. python 3.6x 또는 python 3.7x 버전대 설치
2. 실행에 필요한 라이브러리 설치
```
pip install -r requirements.txt
```
3. tensorflow 설치

 CPU를 사용하는 경우
```
pip install tensorflow==1.13.1    
```
또는 GPU를 사용하는 경우 

```
pip install tensorflow-gpu==1.13.1   
```
GPU를 사용하는 경우 CUDA 9.x 와 CUDNN이 먼저 설치되어 있어야 합니다. 

### 학습 모델
학습된 모델을 아래 링크에서 다운로드 받아서 소스가 있는 디렉토리(model 디렉토리)에 압축을 풉니다.
경로가 아래의 옵션에 설정한 기본값과 다르다면 실행시 경로를 지정해 주어야 합니다.

[학습모델 다운로드](https://drive.google.com/file/d/1-mIAqb6hUSSxKkzClr7oCCfG9MsKuxCn/view?usp=sharing)

## **사용법**
***멀티 이미지 OCR***
```
python subocr.py -i imagepath -o outfile
```
파일명이 타임코드 형식이거나 타임코드 정보를 담고있는 html 파일이 존재하면
srt 자막으로 저장하고, 그렇지 않으면 일반 텍스트 파일로 저장합니다.

VideoSubFinder 프로그램을 이용하면 자막이미지 파일명이 타임코드 형식으로 저장됩니다.
Subtitle Edit에서 sub/idx 파일을 자막이미지과 타임코드 정보를 가진 html 파일로 저장할 수 있습니다.


### 옵션
  -d : 텍스트 검출 모델의 경로. 기본값은 "./model/craft/weight.h5"
  
  -r : 텍스트 인식 모델 경로. 기본값은 "./model/aocr"
  
  -i : 이미지 경로
  
  -o : 출력 파일명. 기본값은 "sub.srt" 

***단일 이미지 OCR***
```
python ocr.py -i imagefile 
```

### 옵션
  -d : 텍스트 검출 모델의 경로. 기본값은 "./model/craft/weight.h5"
  
  -r : 텍스트 인식 모델 경로. 기본값은 "./model/aocr"
  
  -i : 이미지 파일명
  
  -o : OCR 인식결과 이미지 파일명. 값을 저장하지 않으면 OCR 이미지를 저장하지 않는다. 
  
## **학습**
이미 학습된 모델말고 자신만의 데이터로 모델을 학습하고자 하는 경우
아래의 텍스트 검출 및 인식 소스를 다운로드 받아서 학습시키면 됩니다.

텍스트 검출: https://github.com/RubanSeven/CRAFT_keras

텍스트 인식: https://github.com/emedvedev/attention-ocr

