"""
本文件下存放用于计算损失函数的各个函数
"""
import tensorflow as tf
import parameter.net_parameter as para
import numpy as np


def calc_iou(boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
      iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    该函数可以一次性计算所预测的一个batch的样本里的所有边界框的iou
    """
    with tf.variable_scope('iou'):
        # 将边界框的中心坐标形式转换为左上右下形式(x1,y1,x2,y2)
        boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

        boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)

        # 计算相交部分的左上和右下坐标
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])
        # 计算相交部分的面积
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # 计算相并部分的面积
        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]
        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    # 输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def loss_function(predict, annotations):
    """
    predict: 网络输出的预测结果，二维向量，大小为batch_size*(cell_size*cell_size*(5+class_num))
    annotations: 转换得到的样本原始标定
    """
    with tf.variable_scope('loss'):
        # 从原始预测矩阵中抽取出三个预测部分
        # 抽取出类别概率
        predict_calss_prob = tf.reshape(predict[:, :para.boundary_class_prob],
                                        [para.batch_size, para.cell_size, para.cell_size, para.CLASS_NUM])
        # 抽取出边界框置信度
        predict_confidence = tf.reshape(predict[:, para.boundary_class_prob:para.boundary_confidence],
                                        [para.batch_size, para.cell_size, para.cell_size, para.box_per_cell])
        # 抽取出边界框的位置以及形状参数(x, y, w, h)
        predict_boxes = tf.reshape(predict[:, para.boundary_confidence:],
                                   [para.batch_size, para.cell_size, para.cell_size, para.box_per_cell, 4])

        # 对真实标定数据进行解析
        # 每一个边界框的置信度
        response = tf.reshape(annotations[..., 0], [para.batch_size, para.cell_size, para.cell_size, para.box_per_cell])
        # 获得边框的(x,y,w,h)
        boxes = tf.reshape(annotations[..., 1:5], [para.batch_size, para.cell_size, para.cell_size, 1, 4])
        # 在第三维复制box_per_cell次，并转换为比例
        boxes = tf.tile(boxes, [1, 1, 1, para.box_per_cell, 1]) / para.IMAGE_SIZE
        # 类别概率，采用one-hot编码
        class_prob = annotations[..., 5:]

        # 将每一个边界框的位置预测加上该边界框所对应的cell偏置
        offset = np.transpose(np.reshape(np.array([np.arange(para.cell_size)] * para.cell_size * para.box_per_cell),
                                         (para.box_per_cell, para.cell_size, para.cell_size)), (1, 2, 0))
        offset = tf.reshape(tf.constant(offset, dtype=tf.float32), [1, para.cell_size, para.cell_size,
                                                                    para.box_per_cell])
        offset = tf.tile(offset, [para.batch_size, 1, 1, 1])
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))
        predict_boxes_tran = tf.stack(
            [(predict_boxes[..., 0] + offset) / para.cell_size,
             (predict_boxes[..., 1] + offset_tran) / para.cell_size,
             tf.square(predict_boxes[..., 2]),
             tf.square(predict_boxes[..., 3])], axis=-1)

        # 计算预测的边界框与真实标定的iou
        iou_predict_truth = calc_iou(predict_boxes_tran, boxes)

        # 算出有目标的框[BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        # 没有目标的框 [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        boxes_tran = tf.stack(
            [boxes[..., 0] * para.cell_size - offset,
             boxes[..., 1] * para.cell_size - offset_tran,
             tf.sqrt(boxes[..., 2]),
             tf.sqrt(boxes[..., 3])], axis=-1)

        # class_loss
        class_delta = response * (predict_calss_prob - class_prob)
        class_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * para.class_scale

        # object_loss
        object_delta = object_mask * (predict_confidence - iou_predict_truth)
        object_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
            name='object_loss') * para.object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_confidence
        noobject_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
            name='noobject_loss') * para.noobject_scale

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
            name='coord_loss') * para.coord_scale

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)


def my_loss_function(yolo_out, annotations):
    """
    Rewrite the loss_function
    :param yolo_out: 样本经过yolo网络后输出的原始预测结果
    :param annotations: 样本的原始标定
    :return: 样本损失
    """
    with tf.variable_scope('loss'):

        # 对网络输出进行解析



