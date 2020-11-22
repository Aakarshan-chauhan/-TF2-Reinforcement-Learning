import tensorflow as tf
import numpy as np
import random

out = [10,20,30.4, 12.3, 5.3]
s = tf.constant(out, dtype=np.float32)/10.
out = tf.expand_dims(out, 1)
s = tf.expand_dims(s, 1)
print(s)
print(out)

W = tf.Variable(1, trainable=True, name="Weights", dtype=tf.float32)
)
for i in range(100):
	with tf.GradientTape() as tape:

		preds= [W]*s
		error = tf.reduce_mean((preds - out)**2)
	grads = tape.gradient(error, [W])[0].numpy()
	delta_W = 0.01* grads* tf.reduce_mean(preds - out)
	print(f"Delta W = {delta_W}")
	W.assign_add(delta_W)

x = float(input("Enter a number: "))
print(f"Your number is : {x * W}")
