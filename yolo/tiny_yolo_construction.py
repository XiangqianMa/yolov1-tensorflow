import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import parameter.net_parameter as para
import data.extract_tfrecord as extf


# is_training参数用于区分训练还是测试，true为训练．false为测试
def bulid_tiny_yolo_networks(image, output_size, alpha, keep_prob, is_training):
    """
    定义前向传播过程
    :param image:待输入的样本图片
    :param output_size: 网络最终输出向量的大小
    :param alpha: leaky_relu函数的参数
    :param keep_prob: drop_out层的参数
    :param is_training: 区分是否进行训练
    :return: 网络最终的输出
    """
    with tf.variable_scope('yolo'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=leaky_relu(alpha),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                            ):
            net = slim.conv2d(image, 16, 3, 1, padding='SAME', scope='conv_1')
            # 224x224x16
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_2')
            # 112x112x16
            net = slim.conv2d(net, 32, 3, 1, padding='SAME', scope='conv_3')
            # 112x112x32
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_4')
            # 56x56x32
            net = slim.conv2d(net, 64, 3, 1, padding='SAME', scope='conv_5')
            # 56x56x64
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_6')
            # 28x28x64
            net = slim.conv2d(net, 128, 3, 1, padding='SAME', scope='conv_7')
            # 28x28x128
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_8')
            # 14x14x128
            net = slim.conv2d(net, 256, 3, 1, padding='SAME', scope='conv_9')
            # 14x14x256
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_10')
            # 7x7x256
            net = slim.conv2d(net, 512, 3, 1, padding='SAME', scope='conv_11')
            # 7x7x512
            net = slim.max_pool2d(net, 2, 1, padding='SAME', scope='pooling_12')
            # 7x7x512
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_13')
            # 7x7x1024

            # 将上一层输出的张量展平为一维向量
            net = slim.flatten(net, scope='flat_14')
            net = slim.fully_connected(net, 4096, scope='fc_15')
            # 使用dropout避免过拟合
            net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout_16')
            # 最后一层全连接层输出最后的结果［cell_size*cell_size*(5*box_per_cell+class_num)］
            net = slim.fully_connected(net, output_size, activation_fn=None, scope='fc_17')

    return net


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
