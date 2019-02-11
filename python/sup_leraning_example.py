#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wahido
"""


import numpy as np
import scipy as sp
from scipy import stats
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt


sigma =lambda x:  .2+ .1*(1.2+np.sin(6*x)) 
phi = lambda x: np.sin(8*x)

N = 10000
n_batch = 200

x1 = np.random.uniform(0,1,N) 
y1 = phi(x1)  + sigma(x1) * np.random.normal(0,1,x1.shape)

x_p = np.linspace(0,1,100)
y_p = phi(x_p)
y_p_nonoise = phi(x_p)

x_p_f = np.reshape(x_p, [len(x_p),1]  )
y_p_f = np.reshape(y_p, [len(y_p),1]  )


# Construct the tf graph
def phi_nn(x):
    y = tf.layers.dense(x, 20, activation=tf.tanh, use_bias= True)
    y = tf.layers.dense(y, 20, activation=tf.tanh, use_bias= True)
    val = tf.layers.dense(y, 1, activation = None, use_bias= True)
    return val

x_tf = tf.placeholder(tf.float32, shape= [None, 1])
y_tf = tf.placeholder(tf.float32, shape= [None, 1])
phi_val = phi_nn(x_tf)


cost = tf.square(phi_val - y_tf)
cost = tf.reduce_mean(cost)

optim = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        indi_train = np.random.choice( N, n_batch,replace=False)
        x_train = np.reshape(x1[indi_train], [n_batch,1])
        y_train = np.reshape(y1[indi_train], [n_batch,1])

        _, cost_now = sess.run([optim,cost], feed_dict={x_tf: x_train, y_tf: y_train})    
        if i%100 == 0: 
            print('iteration {:10} | current cost: {:10.4f}'.format(i, cost_now))
            # Save training progress

    phi_trained = sess.run(phi_val, feed_dict = {x_tf: x_p_f, y_tf: y_p_f})

plt.plot(x_p, y_p_nonoise, label= "true $\\phi(x)$")
plt.plot(x1[0::100], y1[0::100], '*', label = "Subset of training data (everything)" )

plt.plot(x_p, phi_trained, label ="Trained phi (trained on all data)")
plt.legend()
plt.show()
