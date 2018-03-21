# MNIST
# MNIST 이미지 데이터집합 다운로드
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data

# mnist 데이터 다운로드
(x_tran, y_train),(x_test, y_test) \
    = mnist.load_data()

mnist = input_data.read_data_sets('MNIST/', one_hot=True)



