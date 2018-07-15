import tensorflow as tf
import cv2
import numpy as np
import os
import parameter.net_parameter as para
import data.datasets_preprocess as dp


# 定义函数转化变量类型。在将样本图片及标定数据写入tfrecord文件之前需要对两者的数据类型进行转换
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 数组形式的数据，首先转换为string，再转换为二进制形式进行保存
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord():
    # 创建tfrecord文件
    tfrecord_filename = os.path.join(para.TFRecord_PATH, 'pascal_voc_train.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    # 获取样本名称
    data_path = os.path.join(para.data_path, 'Annotations')
    samples_name = os.listdir(data_path)

    count = 0
    print("Start writing tfrecords.")
    # 写入tfrecord文件
    with tf.Session() as sess:
        for index in samples_name:
            index = index.split('.')[0]
            print("Writing sample:", index)

            # 获得当前样本的标定信息
            label, length = dp.load_pascal_annotation(index)
            # 转换为字符串
            label = label.tostring()

            # 获得样本图片
            img = dp.load_image(index)
            img = img.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'img': _bytes_feature(img),
                'label': _bytes_feature(label)
                }))

            # 将一个example写入tfrecord文件，首先转换为字符串
            writer.write(example.SerializeToString())
            count += 1
    print("Finish,", count, "samples.")


create_tfrecord()