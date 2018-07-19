"""
本文件下存放用于计算损失函数的各个函数
"""
import tensorflow as tf
import parameter.net_parameter as para
import numpy as np
import data.extract_tfrecord as extf
import yolo.yolo_construction as yolo


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


# 网络的输入是以batch的形式进行的，输出同样是以batch的形式进行的．
def my_loss_function(yolo_out, annotations):
    """
    Rewrite the loss_function
    :param yolo_out: 样本经过yolo网络后输出的原始预测结果，为向量　[batch_size, cell_size*cell_size*(5*box_per_cell+class_num)]
    :param annotations: 样本的原始标定，为张量形式　[batch_size, cell_size, cell_size, (5+class_num)]
    :return: 样本损失
    """
    with tf.variable_scope('loss'):
        # 解析出类别概率，这里有个需要注意的细节问题，要使用tf.reshape()，而不是np.reshape．
        predict_class_prob = yolo_out[:, :para.boundary_class_prob]
        predict_class_prob = tf.reshape(predict_class_prob, [para.batch_size, para.cell_size, para.cell_size,
                                                             para.CLASS_NUM])
        # 解析出是否存在目标的置信度
        predict_confidence = yolo_out[:, para.boundary_class_prob:para.boundary_confidence]
        predict_confidence = tf.reshape(predict_confidence, [para.batch_size, para.cell_size, para.cell_size,
                                                             para.box_per_cell])
        # 解析出bounding_box的参数信息，网络预测的bbox的中心坐标是相对于cell的偏移量
        predict_bboxs = yolo_out[:, para.boundary_confidence:]
        predict_bboxs = tf.reshape(predict_bboxs, [para.batch_size, para.cell_size, para.cell_size, para.box_per_cell,
                                                   4])

        # 对真实标定数据的形状进行调整
        true_class_prob = annotations[..., 5:]
        true_confidence = tf.reshape(annotations[..., 0], [para.batch_size, para.cell_size, para.cell_size, 1])
        # 原始标定中的bbox给出的是像素坐标，需要转换为比例（以图像大小为基准）
        true_bboxs = tf.reshape(annotations[..., 1:5], [para.batch_size, para.cell_size, para.cell_size, 1, 4])
        true_bboxs = tf.tile(true_bboxs, [1, 1, 1, para.box_per_cell, 1]) / para.IMAGE_SIZE

        # 将网络所预测的bbox相对于cell的偏移量转换为bbox的中心坐标在图像中的比例
        offset = np.transpose(np.reshape(np.array([np.arange(para.cell_size)] * para.cell_size * para.box_per_cell),
                                         (para.box_per_cell, para.cell_size, para.cell_size)), (1, 2, 0))
        # 转换为四维矩阵
        offset = tf.reshape(tf.constant(offset, dtype=tf.float32), [1, para.cell_size, para.cell_size,
                                                                    para.box_per_cell])
        # 将第０维复制batch_size次
        offset = tf.tile(offset, [para.batch_size, 1, 1, 1])
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))
        # 将中心坐标转换为比例，bbox的长宽进行平方运算，进行平方的原因在于在损失函数中对长宽进行了开方，因而网络预测的是长宽的开方，
        # tf.stack的作用为将数组按照某一维进行堆叠．
        predict_bboxs_tran = tf.stack([(predict_bboxs[..., 0] + offset) / para.cell_size,
                                       (predict_bboxs[..., 1] + offset_tran) / para.cell_size,
                                       tf.square(predict_bboxs[..., 2]),
                                       tf.square(predict_bboxs[..., 3])],
                                      axis=-1)

        # 判断每一个cell中的两个边界框的哪一个负责预测该cell中的目标，与cell中的真实边界框的IOU最大的框负责预测当前目标．
        # 计算预测的边界框与真实标定的iou
        iou_predict_truth = calc_iou(predict_bboxs_tran, true_bboxs)

        # 算出有目标的框[BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        # 计算各个cell各自所预测的几个边界框中的IOU的最大值
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        # 首先得出当前cell中负责进行目标预测的框，再与真实的置信度进行点乘，得出真实的包含有目标的cell中负责进行目标预测的框．
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * true_confidence

        # 没有目标的框 [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        # 将真实bbox的中心坐标转换为相对于cell_box的偏移比例，对长宽进行开方
        boxes_tran = tf.stack([true_bboxs[..., 0] * para.cell_size - offset,
                               true_bboxs[..., 1] * para.cell_size - offset_tran,
                               tf.sqrt(true_bboxs[..., 2]),
                               tf.sqrt(true_bboxs[..., 3])], axis=-1)

        # class_loss
        class_delta = true_confidence * (predict_class_prob - true_class_prob)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta),
                                                  axis=[1, 2, 3]),
                                    name='class_loss') * para.class_scale

        # object_loss
        object_delta = object_mask * (predict_confidence - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta),
                                                   axis=[1, 2, 3]),
                                     name='object_loss') * para.object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_confidence
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta),
                                                     axis=[1, 2, 3]),
                                       name='noobject_loss') * para.noobject_scale

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_bboxs - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta),
                                                  axis=[1, 2, 3, 4]),
                                    name='coord_loss') * para.coord_scale

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)

        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)


if __name__ == '__main__':
    file = 'pascal_voc_train.tfrecords'
    images = tf.placeholder(tf.float32, [None, para.IMAGE_SIZE, para.IMAGE_SIZE, para.IMAGE_CHANNELS])
    labels = tf.placeholder(tf.float32, [None, para.cell_size, para.cell_size, (5+para.CLASS_NUM)])
    output_size = para.cell_size * para.cell_size * (5 * para.box_per_cell + para.CLASS_NUM)
    net_output = yolo.bulid_networks(images, output_size, 0.2, 0.5, True)
    batch_example, batch_label = extf.parse_batch_size_examples(file)
    my_loss_function(net_output, labels)
    total_loss = tf.losses.get_total_loss()

    with tf.Session() as sess:
        inital = tf.global_variables_initializer()
        sess.run(inital)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        example, label = sess.run([batch_example, batch_label])
        print(np.shape(example), np.shape(label))
        feed_dict = {images: example, labels: label.astype(np.float32)}
        output, loss = sess.run([net_output, total_loss], feed_dict=feed_dict)
        print(np.shape(output), loss)

        print(np.shape(output))
        coord.request_stop()
        coord.join(threads)

