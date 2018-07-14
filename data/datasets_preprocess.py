"""
本文件下存放用于对数据集进行预处理的文件，将标准voc格式的数据集处理为yolo网络需要的数据集格式．

"""
import parameter.net_parameter as para
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET


def load_pascal_annotation(index):
    """
    对单个xml文件进行解析
    函数的输入为单个图片样本对应的标定文件
    输出为该图片样本对应的label矩阵，label为三维矩阵[cell_size, cell_size, 5],其中第三维的存储内容为　
    (confidence, x, y, w, h, 20维向量):
        confidence取值方法为:当前单元格包含目标则为１，不包含目标为０；
        (x, y, w, h)为box的形状信息，中心坐标，宽，高，均以像素坐标的形式给出
        20维向量表示当前单元格的目标信息类别，若为第n类目标，则该向量的第n-1个值为１，其余为0，不同的数据集由于类别数目不同，
        20可能取不同的值
    """
    # 读取样本图像，获取其形状信息
    imname = os.path.join(para.data_path, 'JPEGImages', index + '.jpg')
    im = cv2.imread(imname)
    h_ratio = 1.0 * para.IMAGE_SIZE / im.shape[0]
    w_ratio = 1.0 * para.IMAGE_SIZE / im.shape[1]

    # 声明label（三维矩阵），8的计算方法为(4 + 1 + number_class)
    length = 4 + 1 + para.CLASS_NUM
    label = np.zeros((para.cell_size, para.cell_size, length))
    # 解析出标定文件中的object
    filename = os.path.join(para.data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')

    # 对每一个object以此进行解析，label数据的计算是基于原始样本图像的
    for obj in objs:
        bbox = obj.find('bndbox')

        # 将原始样本的标定转换为resize后的图片的标定,按照等比例转换的方式,从0开始索引
        x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, para.IMAGE_SIZE - 1), 0)
        y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, para.IMAGE_SIZE - 1), 0)
        x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, para.IMAGE_SIZE - 1), 0)
        y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, para.IMAGE_SIZE - 1), 0)

        # 将类别由字符串转换为对应的int数
        cls_ind = dict(zip(para.CLASSES, range(len(para.CLASSES))))[obj.find('name').text.lower().strip()]
        # boxes的前两个数据为object的bbox的中心坐标，后两个数据为bbox的width, height
        boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
        # 计算当前目标属于第几个cell，从0开始索引
        x_ind = int(boxes[0] * para.cell_size / para.IMAGE_SIZE)
        y_ind = int(boxes[1] * para.cell_size / para.IMAGE_SIZE)
        # 若(x_ind, y_ind)cell已经标定为存在object,则不写入标定信息，也就是说每一个cell只检测一种目标
        if label[y_ind, x_ind, 0] == 1:
            continue
        # 向(x_ind, y_ind)依次写入是否存在object(对应预测值中的边界框预测置信度)， boxes信息， 类别信息，标定信息的写入顺序要和预测值中
        # 的顺序相同
        label[y_ind, x_ind, 0] = 1
        label[y_ind, x_ind, 1:5] = boxes
        label[y_ind, x_ind, 5 + cls_ind] = 1

    # 返回标定信息，存在object的数目
    return label, len(objs)


def load_image(index):
    """
    依据index对相应的样本图片进行加载，同时执行resize操作，并对图像进行归一化操作
    """
    image_name = os.path.join(para.data_path, 'JPEGImages', index + '.jpg')
    img = cv2.imread(image_name)

    img = cv2.resize(img, (para.IMAGE_SIZE, para.IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, -1)
