
import tensorflow as tf
import numpy as np

def gkernel(x, y, s):
    return tf.divide(1.0,tf.sqrt(tf.multiply(tf.multiply(2.0,np.pi),s))) * tf.exp( tf.divide(tf.multiply(-1.0,tf.pow(tf.subtract(x,y), 2.0)),tf.multiply(2.0,tf.pow(s, 2.0))) )


####### Normalized RKHS Correntropy Layer
def gspace(x,y,s): # x: [bs, nw, sig]
    with tf.name_scope('gspace') as scope:
        def rloop(i):
            return gkernel(tf.gather(x, tf.range(tf.shape(x)[2]), axis=2), tf.expand_dims(tf.gather(y, i, axis=2), dim=2), s)

        return tf.transpose(tf.reduce_mean(tf.map_fn(rloop, tf.range(tf.shape(y)[2]), dtype=tf.float32, parallel_iterations=8), axis=2), [1, 0, 2])


def compute_rkhs(samples, Sigma):

    gsr = tf.image.per_image_standardization(gspace(samples, samples, Sigma))
    return gsr

def correntropy_loss(labels, logits, sigma):
    return tf.reduce_mean(tf.reduce_mean(gkernel(tf.layers.flatten(labels), tf.layers.flatten(logits), sigma), axis=1))
