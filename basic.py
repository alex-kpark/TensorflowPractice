
import tensorflow as tf

# CPU 조건 무시를 위한 소스코드
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
> Building the Graph
- 행렬을 만들 때는, tf.constant()안에 [] 먼저 한 번 써주고 안에서 행렬 정의해야 함
- 아래 연산에는 총 3개의 graph가 생성됨
'''

# 1x2 행렬
matrix1 = tf.constant([[3., 3.]])

# 2x1 행렬
matrix2 = tf.constant([[2.],[2.]])

# 행렬 곱
product = tf.matmul(matrix1, matrix2)

'''
> Launch the graph in a Session
- Graph를 만든 다음에 Session을 만들어서 Graph를 실행해야 한다
'''

# 세션 정의
sess = tf.Session()

# run() 메소드를 호출해서 작업의 결과물인 product의 값을 Return
result = sess.run(product)
print(result)

# 효율적인 시스템 자원 활용을 위해서는 세션을 닫아야 함
with tf.Session() as sess:
    result = sess.run(product)
    print(result)