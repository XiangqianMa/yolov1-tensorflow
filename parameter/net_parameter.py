# 数据集信息
data_path = "/home/mxq/graduation_project/pascal_voc/data/pascal_voc/VOCdevkit/VOC2007"
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
TFRecord_PATH = "/home/mxq/Project/object_detection/yolov1/data/TFRecord"
Tensorboard_PATH = "/home/mxq/Project/object_detection/yolov1/Tensorboard"
TEST_PATH = "/home/mxq/Project/object_detection/yolov1/test/images"
SAVE_PATH = "/home/mxq/Project/object_detection/yolov1/model/"

# 样本参数
CLASS_NUM = 20
IMAGE_SIZE = 448
IMAGE_CHANNELS = 3
SAMPLES_NUM = 5011
# 输出向量参数

# 训练参数
batch_size = 30
base_learn_rate = 0.0005
decay_rate = 0.0005
staircase = True
max_iteration = 10000
alpha = 0.6
keep_prob = 1.0
summary_iteration = 5
save_iteration = 200

# 输出结果解析参数
cell_size = 7
box_per_cell = 2
# 此边界之前的为类别概率，每一个cell包含CLASS_NUM个类别概率
boundary_class_prob = cell_size * cell_size * CLASS_NUM
# 上一边界到该边界之间为所预测的cell包含目标的置信度，每一个cell两个置信度
boundary_confidence = boundary_class_prob + cell_size * cell_size * box_per_cell

# 测试参数
prob_threshold = 0.6
iou_threshold = 0.5

# 损失函数中的权重系数
class_scale = 1.0
object_scale = 1.0
noobject_scale = 0.5
coord_scale = 5.0
