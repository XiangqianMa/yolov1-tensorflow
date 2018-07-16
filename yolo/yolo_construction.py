import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import parameter.net_parameter as para
import data.extract_tfrecord as extf


# 定义前向传播过程
# is_training参数用于区分训练还是测试，true为训练．false为测试
def bulid_networks(image, output_size, alpha, keep_prob, is_training):
    with tf.variable_scope('yolo'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # activation_fn=tf.nn.leaky_relu(alpha=alpha),
                            activation_fn=leaky_relu(alpha),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
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
            net = slim.conv2d(net, 512, 1, 1, padding='SAME', scope='conv_23')
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_24')
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_25')
            net = slim.conv2d(net, 1024, 3, 2, padding='SAME', scope='conv_26')
            # 7x7x1024
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_27')
            net = slim.conv2d(net, 1024, 3, 1, padding='SAME', scope='conv_28')
            # 7x7x1024
            # 将上一层输出的张量展平为一维向量［image_size*image_size*image_channels］
            net = slim.flatten(net, scope='flat_29')
            net = slim.fully_connected(net, 4096, scope='fc_30')
            # 使用dropout避免过拟合
            net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout_31')
            # 最后一层全连接层输出最后的结果［cell_size*cell_size*(5*box_per_cell+class_num)］
            net = slim.fully_connected(net, output_size, activation_fn=None, scope='fc_32')

    return net


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op


file = 'pascal_voc_train.tfrecords'
images = tf.placeholder(tf.float32, [None, para.IMAGE_SIZE, para.IMAGE_SIZE, para.IMAGE_CHANNELS])
output_size = para.cell_size * para.cell_size * (5 * para.box_per_cell + para.CLASS_NUM)
net_output = bulid_networks(images, output_size, 0.2, 0.5, True)
batch_example, batch_label = extf.parse_batch_size_examples(file)

with tf.Session() as sess:
    inital = tf.global_variables_initializer()
    sess.run(inital)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    example, label = sess.run([batch_example, batch_label])
    feed_dict = {images: example}
    output = sess.run(net_output, feed_dict=feed_dict)

    print(np.shape(output))
    coord.request_stop()
    coord.join(threads)



