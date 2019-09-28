import tensorflow as tf
hello = tf.constant("Hello TF world!")
sess = tf.Session()
print(sess.run(hello))
