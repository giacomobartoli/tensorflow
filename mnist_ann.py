
# for osx users: see SOF
# https://stackoverflow.com/questions/49570464/error-importing-mnist-dataset-from-tensorflow-and-ssl-certificate-error-anaconda

import tensorflow as tf
import numpy as np

# -- FASE DI COSTRUZIONE DELLA RETE -----------------------------------------------------------

n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

#funzione per creare dei neuron layers, la sfruttiamo per creare una deep neural network [ci sono cmq funzioni per farlo]
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs) # converge in fretta
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name="weights") # WEIGHTS
            b = tf.Variable(tf.zeros([n_neurons]), name="biases") # a,b = biases
            z = tf.matmul(X, W) + b #moltiplico input x pesi + bias
            if activation=="relu":
                return tf.nn.relu(z)
            else:
                return z
# !POSSO ANCHE USARE FULLY_CONNECTED() DI DEFAULT, sostituendola con neuron_layer
# rete neurale: INPUT LAYER -> 2 HIDDEN LAYER -> OUTPUT

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")

#with tf.name_scope("dnn"):
#    hidden1 = fully_connected(X, n_hidden1, "hidden1", activation="relu")
#    hidden2 = fully_connected(hidden1, n_hidden2, "hidden2", activation="relu")
#    logits = fully_connected(hidden2, n_outputs, "outputs")

#ABBIAMO RETE CON TOPOLOGIA DEEP, manca la funzione di costo.
#Usiamo cross entropy: penalizza i modelli con bassa probabilità per certe classi

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

#MANCA IL GRADIENT DISCENT OPTIMIZIER
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#VALUTIAMO LE PERFORMANCE
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#---FASE DI ESECUZIONE ----------------------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 400
batch_size = 50 #training set size

# FASE DI TRAINING:

with tf.Session() as sess:
    init.run() #inizializzo le variabili
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,y: mnist.test.labels})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt") #salvo i parametri su disco

# FASE DI INFERENCE

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt") #riprendo i parametri appresi
    X_new_scaled = [...] # qui carico nuove immagini da classificare
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1) #prendo solo la classe con prob massima

# CONSIDERAZIONI
# Iperparametri: n neuroni, n livelli, f attivaz, inizializz pesi.. come faccio a fare una buona scelta?
#
# Obviously the number of neurons in the input and output layers is determined by the type of input and output your task requires.
# For example, the MNIST task requires 28 x 28 = 784 input neurons and 10 output neurons. As for the hidden layers, a common practice
# is to size them to form a funnel, with fewer and fewer neurons at each layer— the rationale being that many low-level features can coalesce
# into far fewer high-level features. For example, a typical neural network for MNIST may have two hidden lay‐ ers, the first with 300 neurons and the second with 100.
# However, this practice is not as common now, and you may simply use the same size for all hidden layers—for example, all hidden layers with 150 neurons: that’s just one hyperparameter
# to tune instead of one per layer. Just like for the number of layers, you can try increasing the number of neurons gradually until the network starts overfitting. In general you will get
# more bang for the buck by increasing the number of layers than the number of neurons per layer. Unfortunately, as you can see, finding the perfect amount of neu‐ rons is still somewhat of a black art.
# A simpler approach is to pick a model with more layers and neurons than you actually need, then use early stopping to prevent it from overfitting (and other regu‐ larization techniques, especially dropout,
# as we will see in Chapter 11). This has been dubbed the “stretch pants” approach:12 instead of wasting time looking for pants that perfectly match your size, just use large stretch pants that will shrink down to the right size.
# This has been dubbed the “stretch pants” approach:12 instead of wasting time looking for pants that perfectly match your size, just use large stretch pants that will shrink down to the right size.