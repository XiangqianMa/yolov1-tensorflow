import tensorflow as tf
import numpy as np
import cv2
import os

import yolo.yolo_construction as yolo
import parameter.net_parameter as para


def test_single_image(img):
    """
    本函数用于对单个样本进行检测
    :param img:待检测样本图片
    :return:
    """
    original_img = img
    img_w = np.shape(img)[0]
    img_h = np.shape(img)[1]
    img = cv2.resize(img, (para.IMAGE_SIZE, para.IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, -1)

    output_size = para.cell_size * para.cell_size * (5 * para.box_per_cell + para.CLASS_NUM)
    logits = yolo.bulid_networks(img, output_size, para.alpha, para.keep_prob, False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        yolo_out = sess.run(logits)
        result = interpret_output(yolo_out)

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / para.IMAGE_SIZE)
            result[i][2] *= (1.0 * img_h / para.IMAGE_SIZE)
            result[i][3] *= (1.0 * img_w / para.IMAGE_SIZE)
            result[i][4] *= (1.0 * img_h / para.IMAGE_SIZE)
    draw_result(original_img, result)


def draw_result(img, result):
    for i in range(len(result)):
        x = int(result[i][1])
        y = int(result[i][2])
        w = int(result[i][3] / 2)
        h = int(result[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20),
                      (x + w, y - h), (125, 125, 125), -1)
        lineType = cv2.LINE_AA
        cv2.putText(
            img, result[i][0] + ' : %.2f' % result[i][5],
            (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, lineType)


def interpret_output(yolo_out):
    """
    对网络的输出进行解析，得出样本的检测结果，单样本．
    :param:yolo_out:网络的原始输出
    :return: 检测出的结果[box_num, x, y, w, h, prob]
    """
    # 解析出类别概率，这里有个需要注意的细节问题，要使用tf.reshape()，而不是np.reshape．
    predict_class_prob = yolo_out[0:para.boundary_class_prob]
    predict_class_prob = tf.reshape(predict_class_prob, [para.cell_size, para.cell_size, para.CLASS_NUM])
    # 解析出是否存在目标的置信度
    predict_confidence = yolo_out[para.boundary_class_prob:para.boundary_confidence]
    predict_confidence = tf.reshape(predict_confidence, [para.cell_size, para.cell_size, para.box_per_cell])
    # 解析出bounding_box的参数信息，网络预测的bbox的中心坐标是相对于cell的偏移量
    predict_bboxs = yolo_out[para.boundary_confidence:]
    predict_bboxs = tf.reshape(predict_bboxs, [para.cell_size, para.cell_size, para.box_per_cell, 4])

    # 将网络所预测的bbox相对于cell的偏移量转换为bbox的中心坐标在图像中的比例
    offset = np.array([np.arange(para.cell_size)] * para.cell_size * para.box_per_cell)
    offset = np.transpose(
        np.reshape(
            offset,
            [para.box_per_cell, para.cell_size, para.cell_size]),
        (1, 2, 0))

    # 将中心坐标和宽，长转换为真实的像素值
    predict_bboxs[:, :, :, 0] += offset
    predict_bboxs[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    predict_bboxs[:, :, :, :2] = 1.0 * predict_bboxs[:, :, :, 0:2] / para.cell_size
    predict_bboxs[:, :, :, 2:] = np.square(predict_bboxs[:, :, :, 2:])

    predict_bboxs = predict_bboxs*para.IMAGE_SIZE

    # 计算得出cell中的各个预测框最终给出的概率值，prob=class_prob*confidence
    prob = np.zeros([para.cell_size, para.cell_size, para.box_per_cell, para.CLASS_NUM])
    for box in range(para.box_per_cell):
        for class_n in range(para.CLASS_NUM):
            prob[:, :, box, class_n] = predict_confidence[:, :, box] * predict_class_prob[:, :, class_n]

    # 依据概率和阈值进行过滤
    filter_probs = np.array(prob >= para.prob_threshold, dtype='bool')
    filter_boxes = np.nonzero(filter_probs)

    probs_filtered = prob[filter_probs]
    boxes_filtered = predict_bboxs[filter_boxes[0], filter_boxes[1], filter_boxes[2]]

    classes_num_filtered = np.argmax(
        filter_probs, axis=3)[
        filter_boxes[0], filter_boxes[1], filter_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > para.iou_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append(
            [para.CLASSES[classes_num_filtered[i]],
             boxes_filtered[i][0],
             boxes_filtered[i][1],
             boxes_filtered[i][2],
             boxes_filtered[i][3],
             probs_filtered[i]])

    return result


def iou(box1, box2):
    """
    计算两个Box的iou
    :param box1: box1
    :param box2: box2
    :return: iou
    """
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
         max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
         max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    inter = 0 if tb < 0 or lr < 0 else tb * lr

    return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)


