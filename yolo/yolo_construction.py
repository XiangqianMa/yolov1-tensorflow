import tensorflow as tf
from tensorflow.contrib import slim


# 定义前向传播过程
# is_training参数用于区分训练还是测试，true为训练．false为测试
def bulid_networdks(image, output_size, alpha, keep_prob, is_training):
    with tf.variable_scope('yolo'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn = tf.nn.leaky_relu(alpha=alpha),
                            weights_regularizer = slim.l2_regularizer(0.0005),
                            weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)
                            ):
            net = slim.conv2d(image, 64, 7, 2, padding='SAME', scope='conv_1')
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_2')
            # 112x112x64
            net = slim.conv2d(net, 192, 3, 1, padding='SAME', scope='conv_3')
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_4')
            # 56x56x192
            net = slim.conv2d(net, 128, 1, 1, padding='SAME', scope='conv_5')
            net = slim.conv2d(net, 256, 3, 1, padding='SAME', scope='conv_6')
            net = slim.conv2d(net, 256, 1, 1, padding='SAME', scope='conv_7')
            net = slim.conv2d(net, 512, 3, 1, padding='SAME', scope='conv_8')
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_9')
            # 28x28x512
            # 结构重复４次
            net = slim.conv2d(net, 256, 1, 1, padding='SAME', scope='conv_10')
            net = slim.conv2d(net, 512, 3, 1, padding='SAME', scope='conv_11')
            net = slim.conv2d(net, 256, 1, 1, padding='SAME', scope='conv_12')
            net = slim.conv2d(net, 512, 3, 1, padding='SAME', scope='conv_13')
            net = slim.conv2d(net, 256, 1, 1, padding='SAME', scope='conv_14')
            net = slim.conv2d(net, 512, 3, 1, padding='SAME', scope='conv_15')
            net = slim.conv2d(net, 256, 1, 1, padding='SAME', scope='conv_16')
            net = slim.conv2d(net, 512, 3, 1, padding='SAME', scope='conv_17')

            net = slim.conv2d(net, 512, 1, 1, padding='SAME', scope='conv_18')
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_19')
            net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pooling_20')
            # 14x14x1024
            # 结构重复２次
            net = slim.conv2d(net, 512, 1, 1, padding='SAME', scope='conv_21')
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_22')
            net = slim.conv2d(net, 512, 1, 1, padding='SAME', scope='conv_22')
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_23')
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_24')
            net = slim.conv2d(net, 1024, 3, 2, padding='SAME', scope='conv_25')
            # 7x7x1024
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_26')
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_27')
            # 7x7x1024
            # 将上一层输出的张量展平为一维向量［batch_size, image_size*image_size*image_channels］
            net = slim.flatten(net, scope='flat_28')
            net = slim.fully_connected(net, 4096, scope='fc_29')
            # 使用dropout避免过拟合
            net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout_30')
            # 最后一层全连接层输出最后的结果［cell_size*cell_size*(5+class_num)］
            net = slim.fully_connected(net, output_size, activation_fn=None, scope='fc_31')

    return net


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op



