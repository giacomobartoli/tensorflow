# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
# import os

# Create a graph
import tensorflow as tf

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, n = housing.data.shape

# Gradient Discent requires to scale vectors first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000  # not shown in the book
learning_rate = 0.01  # not shown

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")  # not shown
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")  # not shown
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")  # not shown
error = y_pred - y  # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")  # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # not shown
training_op = optimizer.minimize(mse)  # not shown

# Once you have trained your model, you should save its parameters to disk so you can come back to it whenever you want,
# use it in another program, compare it to other models, and so on.
# Moreover, you probably want to save checkpoints at regular intervals during training so that if your computer crashes
# during training you can continue from the last checkpoint rather than start over from scratch.
# TensorFlow makes saving and restoring a model very easy.
# Just create a Saver node at the end of the construction phase (after all variable nodes are created);
# then, in the execution phase, just call its save() method whenever you want to save the model,
# passing it the session and path of the checkpoint file:


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:  # checkpoint every 100 epochs
            print("Epoch", epoch, "MSE =", mse.eval())
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
    best_theta

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")  # restoring the model
    best_theta_restored = theta.eval()