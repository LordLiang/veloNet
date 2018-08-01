import tensorflow as tf

from config import cfg
from utils.op import *

class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
        self.training = training
        self.batch_size = batch_size

        self.input = tf.placeholder(
            tf.float32, [self.batch_size, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, cfg.CHANNEL_SIZE], name='feature')

        with tf.variable_scope('FeatureNet_' + name):
            # convolutinal middle layers
            temp_conv = ConvMD(10, 32, 3, (2, 2), (1, 1),
                               self.input, training=self.training, name='conv1')
            temp_conv = ConvMD(32, 64, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv2')
            temp_conv = ConvMD(64, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv3')

            self.outputs = temp_conv