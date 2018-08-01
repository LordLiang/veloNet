import tensorflow as tf
import numpy as np

from config import cfg
from utils.op import *

small_addon_for_BCE = 1e-6

class RPN(object):
    def __init__(self, input, alpha=1.5, beta=1, sigma=3, training=True, name=''):
        # scale = [batchsize, 800, 704, 10]
        self.input = input
        self.training = training
        # groundtruth(target) - each anchor box, represent as ¡÷x, ¡÷y, ¡÷z, ¡÷l, ¡÷w, ¡÷h, rotation
        self.targets = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 28])
        # postive anchors equal to one and others equal to zero(4 anchors in 1 position)
        self.pos_equal_one = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 4])
        self.pos_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])
        self.pos_equal_one_for_reg = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 28])
        # negative anchors equal to one and others equal to zero
        self.neg_equal_one = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 4])
        self.neg_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])

        with tf.variable_scope('RPN_' + name):
            # block1:
            temp_conv = ConvMD(128, 128, 3, (2, 2), (1, 1),
                               self.input, training=self.training, name='conv1')
            temp_conv = ConvMD(128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv2')
            temp_conv = ConvMD(128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv3')
            temp_conv = ConvMD(128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv4')
            deconv1 = Deconv2D(128, 256, 3, (1, 1), (0, 0),
                               temp_conv, training=self.training, name='deconv1')
            # print('deconv1', deconv1.shape)#batchsize, 200, 176, 256

            # block2:
            temp_conv = ConvMD(128, 128, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv5')
            temp_conv = ConvMD(128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv6')
            temp_conv = ConvMD(128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv7')
            temp_conv = ConvMD(128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv8')
            temp_conv = ConvMD(128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv9')
            temp_conv = ConvMD(128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv10')
            deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0),
                               temp_conv, training=self.training, name='deconv2')
            # print('deconv2', deconv2.shape)#batchsize, 200, 176, 256

            # block3:
            temp_conv = ConvMD(128, 256, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv11')
            temp_conv = ConvMD(256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv12')
            temp_conv = ConvMD(256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv13')
            temp_conv = ConvMD(256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv14')
            temp_conv = ConvMD(256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv15')
            temp_conv = ConvMD(256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv16')
            deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0),
                               temp_conv, training=self.training, name='deconv3')
            # print('deconv3', deconv3.shape)#batchsize, 200, 176, 256

            # final:
            temp_conv = tf.concat([deconv3, deconv2, deconv1], -1)

            # Probability score map, scale = [None, 200, 176, 4]
            p_map = ConvMD(768, 4, 1, (1, 1), (0, 0), temp_conv,
                           training=self.training, activation=False, bn=False, name='conv17')
            # print('p_map', p_map.shape)
            # Regression(residual) map, scale = [None, 200, 176, 28]
            r_map = ConvMD(768, 28, 1, (1, 1), (0, 0),
                           temp_conv, training=self.training, activation=False, bn=False, name='conv18')
            # print('r_map', r_map.shape)
            # softmax output for positive anchor and negative anchor, scale = [None, 200/100, 176/120, 1]
            self.p_pos = tf.sigmoid(p_map)
            # self.p_pos = tf.nn.softmax(p_map, dim=3)
            self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]# 200, 176

            self.cls_pos_loss = (-self.pos_equal_one * tf.log(self.p_pos + small_addon_for_BCE)) / self.pos_equal_one_sum
            self.cls_neg_loss = (-self.neg_equal_one * tf.log(1 - self.p_pos + small_addon_for_BCE)) / self.neg_equal_one_sum

            self.cls_loss = tf.reduce_sum(alpha * self.cls_pos_loss + beta * self.cls_neg_loss)
            self.cls_pos_loss_rec = tf.reduce_sum(self.cls_pos_loss)
            self.cls_neg_loss_rec = tf.reduce_sum(self.cls_neg_loss)

            self.reg_loss = smooth_l1(r_map * self.pos_equal_one_for_reg, self.targets *
                                      self.pos_equal_one_for_reg, sigma) / self.pos_equal_one_sum
            self.reg_loss = tf.reduce_sum(self.reg_loss)

            self.loss = tf.reduce_sum(self.cls_loss + self.reg_loss)

            self.delta_output = r_map
            self.prob_output = self.p_pos

