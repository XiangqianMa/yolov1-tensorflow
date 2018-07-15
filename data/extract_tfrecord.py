import tensorflow as tf
import cv2
import numpy as np
import os
import parameter.net_parameter as para


def parse_single_example(file_name):
    """
    :param file_name:待解析的tfrecord文件的名称
    :return: 从文件中解析出的单个样本的相关特征，img, label
    """
    tfrecord_filename = os.path.join(para.TFRecord_PATH, file_name)

    # 定义解析TFRecord文件操作
    reader = tf.TFRecordReader()
    # 创建样本文件名称队列
    filename_queue = tf.train.string_input_producer([tfrecord_filename])

    # 解析单个样本文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    img = features['img']
    label = features['label']

    return img, label


def parse_batch_size_examples(file_name):
    """
    :param file_name:待解析的tfrecord文件的名称
    :return: 解析得到的batch_size个样本
    """
    batch_size = para.batch_size
    min_after_dequeue = 10
    num_threads = 3
    capacity = min_after_dequeue + 3 * batch_size

    image, label = parse_single_example(file_name)
    print('hello')
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=num_threads,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue
                                                      )

    image_batch = tf.decode_raw(image_batch, tf.float32)
    label_batch = tf.decode_raw(label_batch, tf.float64)
    print('enen')
    return image_batch, label_batch


file = 'pascal_voc_train.tfrecords'
batch_example, batch_label = parse_batch_size_examples(file)
with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    example, label = sess.run([batch_example, batch_label])

    print(type(batch_example))
    # cv2.imshow('img', example)
    # cv2.waitKey(0)
    # print(type(example))

    coord.clear_stop()
    coord.join(threads)
