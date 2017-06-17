from elbow import Gaussian, Model
import numpy as np
import tensorflow as tf
x = np.random.random(100)
y = 2*x +.01*np.random.random(100)
# x = tf.constant(x, dtype=tf.float32)

mu = Gaussian(mean=0, std=10, name="mu")
z = x*mu ## unsuported operation
Y = Gaussian(mean=z, std=1, shape=(100,), name="Y")

m = Model(Y) # automatically includes all ancestors of X
sampled = m.sample()

print("mu is", sampled["mu"])
print("empirical mean is", np.mean(sampled["X"]))


X.observe(sampled["X"])
mu.attach_q(Gaussian(shape=mu.shape, name="q_mu"))


m.train(steps=500)
posterior = m.posterior()
print("posterior on mu has mean %.2f and std %.2f" % (posterior["q_mu"]["mean"], posterior["q_mu"]["std"]))

posterior['q_mu']

posterior['observed_X']

