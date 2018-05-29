import tensorflow as tf #importing the tensorflow library
T, F = 1.0, -1.0   #True has the +1.0 value and False has -1.0, it's important to note that
# you can assign any value to them
bias = 1.0
training_input = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]
# OR logic operator
training_output = [
    [T],
    [T],
    [T],
    [F],
]
# The values are initialized to normally-distributed random numbers, for that reason we used tf.random_normal module.
w = tf.Variable(tf.random_normal([3, 1]), dtype=tf.float32)
# step(x) = { 1 if x > 0; -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)
# simply define output, Error and Mean squared error in three lines.
output = step(tf.matmul(training_input, w))
error = tf.subtract(training_output, output)
mse = tf.reduce_mean(tf.square(error))
# We computed the desired adjustment (delta) based on the error we earlier computed. And then proceed to add it to our weights.
delta = tf.matmul(training_input, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))
# The model has to be evaluated by a TensorFlow session, which we instantiate before initializing all variables to their specified values
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# We can now run our model through training epochs, adjusting the weights each time by evaluating train. Since we’re using a binary output, we can expect to reach a perfect result with a mean squared error of 0.
err, target = 1, 0
epoch, max_epochs = 0, 10
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
print('epoch:', epoch, 'mse:', err)

print(sess.run(w))