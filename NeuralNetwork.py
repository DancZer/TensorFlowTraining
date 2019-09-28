import tensorflow as tf
from datetime import datetime

# start board
# tensorboard --logdir=summary_log --port 6006

inputLayerSize = 2
hiddenLayerSize = 3
outputLayerSize = 1

# Boundaries
Input = tf.placeholder(dtype=tf.float32, shape=[None, inputLayerSize], name="Input")
Target = tf.placeholder(dtype=tf.float32, shape=[None, outputLayerSize], name="Target")

# network
inputWeights = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[inputLayerSize, hiddenLayerSize], stddev=0.4), name="InputWeights")
hiddenBias = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[hiddenLayerSize], stddev=0.4), name="HiddenBias")
hiddenLayer = tf.sigmoid(tf.matmul(Input, inputWeights) + hiddenBias, name="hidden_layer_activation")

hiddenWeights = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[hiddenLayerSize, outputLayerSize], stddev=0.4), name="HiddenWeights")
outputBias = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[outputLayerSize], stddev=0.4), name="OutputBias")
Output = tf.sigmoid(tf.matmul(hiddenLayer, hiddenWeights) + outputBias, name="output_layer_activation")

# cost function for training
cost = tf.reduce_mean(tf.squared_difference(Target, Output))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# create summary
tf.summary.histogram(name="InputWeight", values=inputWeights)
tf.summary.histogram(name="HiddenWeight", values=hiddenWeights)
tf.summary.scalar("error", cost)

trainingInput = [[1, 1], [1, 0], [0, 1], [0, 0]]
expectedOutput = [[0], [1], [1], [0]]
epochs = 8000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    mergedSummary = tf.summary.merge_all()
    fileName = "./summary_log/run"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = tf.summary.FileWriter(fileName, sess.graph)
    for e in range(epochs):
        err, _, summaryOutput = sess.run([cost, optimizer, mergedSummary], feed_dict={Input: trainingInput, Target: expectedOutput})
        print(e, err)
        writer.add_summary(summaryOutput, e)
    #
    # while True:
    #     userInput = [[0, 0]]
    #     userInput[0][0] = int(input("Type first input:"))
    #     userInput[0][1] = int(input("Type second input:"))
    #     print("Input is: "+str(userInput))
    #     print(sess.run([Output], feed_dict={Input: userInput})[0][0])
