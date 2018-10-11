#https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/pros/

#Data Download
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

'''
mnist.train -> training data 55,000개
mnist.text -> test data 10,000개
mnist.validation -> validation data 5,000개

X Label(이미지) : xs -> mnist.train.images
Y Label(숫자 구분) : ys -> mnist.train.labels
이미지 크기 : 28x28 = 784

트레이닝에 쓰이는 X Label Data는 [55000, 784] 형태의 Array (55,000개의 데이터 , 784픽셀)
트레이닝에 쓰이는 Y Label Data는 [55000, 10] 형태의 Array (55,000개의 데이터, 10개 라벨)
-> 라벨 3은 [0,0,0,1,0,0,0,0,0,0] 으로 표현 가능

MNIST를 진행하는 CNN에서, 각 픽셀의 어두운 정도(intensity)를 Convolution해서 이미지가 어떤 숫자인지 구분하는 것이다.
-> 벡터공간 개념으로 생각해보면, 숫자마다 가지고 있는 일정한 벡터위치가 있을 것인데, 그 점들이 찍혀있는 곳에 가까운 것으로 결과값을 결정하는 것이다.
-> 데이터와 독립적일 수 있는 쓸데없는 정보일 수도 있으므로 Bias를 두는 것
'''

# Tensorflow에서 Python으로 돌아오는 방식으로 연산을 하면 메모리 소모가 심하기 때문에 tensorflow에 맞는 데이터 형식을 정의함
x = tf.placeholder(tf.float32, [None, 784]) #None은 어떠한 값이 들어올 수 있음을 의미한다

# Empty placeholder 정의 (실제 데이터가 들어갈 값)
y_ = tf.placeholder(tf.float32, [None, 10])

# zeros를 통해 Weight과 bias를 0으로 초기화해주고 시작함
W = tf.Variable(tf.zeros([784, 10])) #784 픽셀을 넣어서 10개의 결과를 만들어 낼 것이므로 [784,10]
b = tf.Variable(tf.zeros([10]))

# 모델 (예측한 값)
y = tf.nn.softmax(tf.matmul(x,W)+b)

'''
학습 정의
- 모델의 손실(cost)를 정의하는데 자주 사용하는 개념은 Cross-Entropy
- 실제 분포를 예측하는데 모델이 얼마나 비효율적인지를 알려주는 값
'''

# Cross Entropy
# logits는 가설(hypothesis)을 의미, labels는 예측값
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) #함수를 그대로 구현 : 실제값 x log(예측값)

# BackPropagation Algorithm의 구현
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #learning_rate = 0.5

# 학습 들어가기 전에 초기화
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init) #초기화 graph 세션에서 실행시키고

# 직접 학습을 시킴
for i in range(1000):
    batch = mnist.train.next_batch(100) #100개씩 training하는 데이터에서 랜덤으로 뽑아서 batch를 형성해줌 -> x, y array로 구성되어있음
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]}) #train_step(경사하강법 쓰는 부분)에 x, y 데이터를 넣어줌

    # 학습 완료 이후 모델의 정확도를 체크
    #argmax는 가장 큰 인덱스를 찾기에 유용한 함수, argmax(y / y_ ,1)는 예측 라벨과 실제 데이터 라벨이 같은지를 체크해줌 --> Boolean 값을 리턴함
    correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32)) # Bool 값을 0,1로 환사내서 평균을 구해줌

    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))