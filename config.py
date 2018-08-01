from easydict import EasyDict as edict

__C = edict()

cfg = __C

# for dataset dir
__C.DATA_DIR = '/adata/zhoujie/KITTI/for_velonet'
__C.IMG_DIR = '/adata/zhoujie/KITTI/object/training/image_2'
__C.CALIB_DIR = '/adata/zhoujie/KITTI/object/training/calib'

# for gpu allocation
__C.GPU_AVAILABLE = '0,1'
__C.GPU_USE_COUNT = len(__C.GPU_AVAILABLE.split(','))
__C.GPU_MEMORY_FRACTION = 1

# selected object
__C.DETECT_OBJ = 'Car'  # Pedestrian/Cyclist
if __C.DETECT_OBJ == 'Car':
    __C.Z_MIN = -3
    __C.Z_MAX = 1
    __C.Y_MIN = -40
    __C.Y_MAX = 40
    __C.X_MIN = 0
    __C.X_MAX = 70.4
    __C.GRID_RESOLUTION = 0.1

    __C.SLICE_SIZE = 8

    __C.INPUT_WIDTH = int((__C.X_MAX - __C.X_MIN)/__C.GRID_RESOLUTION)# x 704
    __C.INPUT_HEIGHT = int((__C.Y_MAX - __C.Y_MIN)/__C.GRID_RESOLUTION)# y 800
    __C.CHANNEL_SIZE = 10

    __C.FEATURE_RATIO = 4
    __C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO)  # 176
    __C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO)# 200



# Faster-RCNN/SSD Hyper params
if __C.DETECT_OBJ == 'Car':
    # car anchor
    __C.ANCHOR_L = 3.9  # l x
    __C.ANCHOR_W = 1.6  # w y
    __C.ANCHOR_H = 1.56 # h z
    __C.ANCHOR_Z = cfg.ANCHOR_H/2 #0.78
    __C.RPN_POS_IOU = 0.6
    __C.RPN_NEG_IOU = 0.45

# set the log image scale factor
__C.BV_LOG_FACTOR = 4

# for data set type
__C.DATA_SETS_TYPE = 'kitti'

# for camera and lidar coordination convert
if __C.DATA_SETS_TYPE == 'kitti':
    # cal mean from train set
    __C.MATRIX_P2 = ([[719.787081,    0.,            608.463003, 44.9538775],
                      [0.,            719.787081,    174.545111, 0.1066855],
                      [0.,            0.,            1.,         3.0106472e-03],
                      [0.,            0.,            0.,         0]])

    # cal mean from train set
    __C.MATRIX_T_VELO_2_CAM = ([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ])
    # cal mean from train set
    __C.MATRIX_R_RECT_0 = ([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ])

# for data preprocess
# rgb
if __C.DATA_SETS_TYPE == 'kitti':
    __C.IMAGE_WIDTH = 1242
    __C.IMAGE_HEIGHT = 375
    __C.IMAGE_CHANNEL = 3

# set the log image scale factor
__C.BV_LOG_FACTOR = 2

# for rpn nms
__C.RPN_NMS_POST_TOPK = 20
__C.RPN_NMS_THRESH = 0.1
__C.RPN_SCORE_THRESH = 0.96