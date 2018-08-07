import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import cv2
import os
from datetime import datetime

import data.extract_tfrecord as extf
import parameter.net_parameter as para
import yolo.yolo_construction as yolo
import yolo.tiny_yolo_construction as tiny_yolo

from yolo import loss_function


def train(tfrecord_file, max_iteration, base_learning_rate, alpha, keep_prob):
    is_training = True
    tensorboard_file = os.path.join(para.Tensorboard_PATH, datetime.now().strftime('%Y_%m_%d_%H_%M'))
    # 计算网络输出大小
    output_size = para.cell_size * para.cell_size * (5 * para.box_per_cell + para.CLASS_NUM)
    # 解析得到训练样本以及标定
    batch_example, batch_label = extf.parse_batch_size_examples(tfrecord_file)

    # 设置输入占位符
    images = tf.placeholder(tf.float32, [None, para.IMAGE_SIZE, para.IMAGE_SIZE, para.IMAGE_CHANNELS])
    labels = tf.placeholder(tf.float32, [None, para.cell_size, para.cell_size, (5+para.CLASS_NUM)])
    # 构建yolo网络
    # net_output = yolo.bulid_networks(images, output_size, alpha, keep_prob, is_training)
    net_output = tiny_yolo.bulid_tiny_yolo_networks(images, output_size, alpha, keep_prob, is_training)
    # 得到损失函数
    loss_function.my_loss_function(net_output, labels)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('total_loss', total_loss)
    # 设置指数衰减学习率
    global_step = tf.train.create_global_step()
    decay_steps = para.SAMPLES_NUM/para.batch_size
    learning_rate = tf.train.exponential_decay(base_learning_rate, global_step, decay_steps,
                                               para.decay_rate, para.staircase)
    # 设置优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    # 设置训练操作
    # train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    # 配置tensorboard
    summary_merge = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(tensorboard_file, flush_secs=60)

    # 模型保存
    save_variable = tf.global_variables()
    saver = tf.train.Saver(save_variable, max_to_keep=True)

    # 配置GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        summary_writer.add_graph(sess.graph)
        for iteration in range(max_iteration-1):
            # 获取样本数据
            example, label = sess.run([batch_example, batch_label])
            feed_dict = {images: example, labels: label.astype(np.float32)}
            # cv2.imshow("img", example[0])
            # cv2.imshow("img1", example[1])
            # cv2.imshow("img2", example[2])
            # cv2.waitKey(500)
            print("Start training:", iteration+1, "iter")

            _, loss, current_learning_rate, current_global_step = sess.run([train_op, total_loss, learning_rate, global_step], feed_dict=feed_dict)

            if iteration % para.summary_iteration == 0:
                summary_str = sess.run(summary_merge, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=iteration)

            if iteration+1 % para.save_iteration == 0:
                print('save ckpt')
                saver.save(sess, para.SAVE_PATH + 'yolov1.ckpt', global_step=current_global_step)

            print("Global_step", current_global_step, "Loss is:", loss, "Learning_rate is:", current_learning_rate)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    file = 'pascal_voc_train.tfrecords'
    max_iter = para.max_iteration
    base_learning_rate = para.base_learn_rate
    alpha = para.alpha
    keep_prob = para.keep_prob

    train(file, max_iter, base_learning_rate, alpha, keep_prob)
