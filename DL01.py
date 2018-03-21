# DL01
# 딥러닝 기초 - 텐서플로, 케라스

# 딥러닝 개발환경
# 1. 파이썬 3.5이상 또는 아나콘다 4.4이상
# 2. 파이참 - 파이썬 venv 환경에서 실행하도록 설정
# 3. 머신러닝/딥러닝 관련 패키지 설치
# numpy scipy matplotlib pandas scikit-learn
# spyder(과학계산) seaborn(시각화) h5py(hdf5) pillow(이미지)
# tensorflow (tensorflow-gpu) keras

# 4. 텐서플로우 GPU 지원 사이트
# CUDA 설치 - https://developer.nvidia.com/cuda-downloads
# cuDNN 설치 - https://developer.nvidia.com/cudnn

# 5. 설치 확인
# 파이참 - '파이썬 콘솔'에서 다음 실행
import tensorflow as tf
print(tf.__version__) # 텐서플로우 버전 확인

import keras

hello = tf.constant('Hello, Tensorflow!!')
sess = tf.Session() # 텐서플로우 작업 생성
print( sess.run(hello) ) # 텐서플로우 작업 실행

# 인공지능 - 관념적으로 컴퓨터가 인간이 사고를 모방하는 것
#           즉, 기계가 인간처럼 사고하고 행동하게 하는 것

# 머신러닝 - 주어진 데이터를 통해 컴퓨터가 스스로 학습하는 것
#           학습 : 데이터를 입력해서 패턴을 분석하는 과정
# 머신러닝의 한계 - 인간도 파악하기 어려운 복잡한 문제는
#                 머신러닝으로도 풀기 어려움 (이미지 인식)

# 딥러닝 - 인공신경망을 이용해서 컴퓨터가 스스로 학습하는 것
#         인공신경망 : 인간 뇌의 동작방식을 착안해서 만듦
# 2012년 ImageNet에서 제공하는 1,000개의 카테고리로
# 분류된 100만개의 이미지를 인식하여 정확성을 겨루는
# ILSVRC라는 이미지 인식대회에서 84.7%라는 인식률을 달성
# 그 전까지는 75%대였음 - 현재는 97%에 육박할 정도

# 인공신경망은 이미 1940년대 부터 연구되던 기술
# 그전까지는 여러가지 문제에 부딪혀 암흑의 시대를 지나다가
# 빅데이터와 GPGU의 발전, 수백만에서 수조개로 이뤄진 아주
# 간단한 수식을 효율적으로 실행하게 해주는 딥러닝 알고리즘의
# 발명 덕택에 급격히 발전되고 있음

# 텐서플로우는 머신러닝 프로그램, 특히 딥러닝 프로그램을
# 아주 쉽게 구현할 수 있도록 다양한 기능을 제공하는 라이브러리
# 구글에서 제작하고 베포하고 있음

# 케라스, 카페, 토치, MXNet, 체이너, CNTK
# 텐서플로우를 좀 더 사용하기 쉽게 만들어 주는 보조 라이브러리



