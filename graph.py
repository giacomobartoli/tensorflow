# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
# import numpy as np
# import os

# Create a graph
import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

# to evaluate a graph I need to open a session, initialize variables and evaluate f
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

# or in a faster way:
# PS: identation determines the scope in Python!
# PS2: the session here is automatically closed
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

# Interactive sessions and global initializer
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()


# A TensorFlow program is typically split into two parts: the first part builds a computation graph
# (this is called the construction phase), and the second part runs it (this is the execution phase).

