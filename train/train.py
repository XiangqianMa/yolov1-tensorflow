import tensorflow as tf
import numpy as np

import data.extract_tfrecord as extf
import parameter.net_parameter as para
import yolo.yolo_construction as yolo

from tensorflow.contrib import slim
from yolo import loss_function


def train(tfrecord_file, max_iteration, base_learning_rate, alpha, keep_prob):
    is_training = True
    # 计算网络输出大小
    output_size = para.cell_size * para.cell_size * (5 * para.box_per_cell + para.CLASS_NUM)
    # 解析得到训练样本以及标定
    batch_example, batch_label = extf.parse_batch_size_examples(tfrecord_file)

    global_step = tf.train.get_global_step()
    # 设置输入占位符
    images = tf.placeholder(tf.float32, [None, para.IMAGE_SIZE, para.IMAGE_SIZE, para.IMAGE_CHANNELS])
    labels = tf.placeholder(tf.float32, [None, para.cell_size, para.cell_size, (5+para.CLASS_NUM)])
    # 构建yolo网络
    net_output = yolo.bulid_networks(images, output_size, alpha, keep_prob, is_training)
    # 得到损失函数
    loss_function.my_loss_function(net_output, labels)
    total_loss = tf.losses.get_total_loss()
    # 设置优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_learning_rate)
    # 设置训练操作
    train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)

    # 配置GPU
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for iteration in range(max_iteration):
            # 获取样本数据
            example, label = sess.run([batch_example, batch_label])
            feed_dict = {images: example, labels: label.astype(np.float32)}

            output, loss = sess.run([total_loss, train_op], feed_dict=feed_dict)
            print(loss)

        coord.request_stop()
        coord.join(threads)