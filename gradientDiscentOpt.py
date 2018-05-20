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


n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# TensorFlow computes the gradients for you.
# But it gets even easier: it also provides a number of optimizers out of the box, including a Gradient Descent optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# TensorFlow’s autodiff feature it can automatically and efficiently compute the gradients for you.
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)