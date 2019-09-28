import tensorflow as tf

input1 = tf.placeholder(tf.float32, [2], "Input1")
input2 = tf.placeholder(tf.float32, [2], "Input2")

output = tf.pow(tf.add(input1, input2), 2)

with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1: [2, 3], input2: [4, 5]})
    print(result)
