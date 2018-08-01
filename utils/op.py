import numpy as np
import tensorflow as tf


def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


def ConvMD(Cin, Cout, k, s, p, input, training=True, activation=True, bn=True, name='conv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    with tf.variable_scope(name) as scope:
        paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
        pad = tf.pad(input, paddings, "CONSTANT")
        temp_conv = tf.layers.conv2d(
            pad, Cout, k, strides=s, padding="valid", reuse=tf.AUTO_REUSE, name=scope)

        if bn:
            temp_conv = tf.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv


def Deconv2D(Cin, Cout, k, s, p, input, training=True, bn=True, name='deconv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    pad = tf.pad(input, paddings, "CONSTANT")
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d_transpose(
            pad, Cout, k, strides=s, padding="SAME", reuse=tf.AUTO_REUSE, name=scope)
        if bn:
            temp_conv = tf.layers.batch_normalization(
                temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        return tf.nn.relu(temp_conv)


def average_gradients(tower_grads):
    # ref:
    # https://github.com/tensorflow/models/blob/6db9f0282e2ab12795628de6200670892a8ad6ba/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L103
    # but only contains grads, no vars
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        grad_and_var = grad
        average_grads.append(grad_and_var)
    return average_grads