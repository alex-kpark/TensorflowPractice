#Data Download
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# CPU 조건 무시를 위한 소스코드
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Learning에 필요한 Paramenter
learning_rate = 0.01
training_epochs = 20
batch_size = 100

# 실제 데이터를 넣을 Placeholder 정의
X = tf.placeholder(tf.float32, [None, 784]) #None은 어떠한 값이 들어올 수 있음을 의미한다
Y = tf.placeholder(tf.float32, [None, 10])

# 가설 식 정의
W = tf.Variable(tf.random_normal([784,10])) #784 픽셀을 넣어서 10개의 결과를 만들어 낼 것이므로 [784,10]
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(X, W) + b

'''
학습 정의
- 모델의 손실(cost)를 정의하는데 자주 사용하는 개념은 Cross-Entropy
- 실제 분포를 예측하는데 모델이 얼마나 비효율적인지를 알려주는 값
'''

# Cost 정의 (Cross-Entropy)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)) #logits에는 예측값(가설), labels에는 실제 데이터를

# Optimizer(학습) 정의
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습 시작
# Epoch를 돌리기 시작
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size) #데이터 전체 갯수를 배치 크기로 나누어서 전체 배치 개수를 구함

    # Batch를 돌리기 시작
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) #애초에 Tuple형으로 데이터를 가져옴
        feed_dict = {X: batch_xs, Y: batch_ys}
        added_cost, _ = sess.run([cost, optimizer], feed_dict=feed_dict) #sess,run()에서 2가지 그래프를 실행시키므로, 각각의 것을 할당해주어야 하나 optimizer는 필요가 없어 _에 할당함
        avg_cost += added_cost / total_batch
    
    print('Epoch:', epoch, 'cost =', avg_cost)

print("### Learning Finished ###") 

#학습 결과 테스팅
correction_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)) #hypothesis와 실제 데이터의 차이를 체크해서 Boolean 형태로 제공
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32)) 

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))