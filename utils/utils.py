import numpy as np
import tensorflow as tf

from config import cfg
from utils.rotate_iou import box2d_rotate_iou


def camera_to_lidar(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.MATRIX_T_VELO_2_CAM)

    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.MATRIX_R_RECT_0)

    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(R_RECT_0), p)
    p = np.matmul(np.linalg.inv(T_VELO_2_CAM), p)
    p = p[0:3]
    return tuple(p)


def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
        numpy array. Returns the 2D projection of the points that
        are in front of the camera only an the corresponding 3D points.'''
    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
    # Before projecting, keep only points with z >= 0
    # (points that are in front of the camera).
    idx = (pts3d_cam[2, :] >= 0)
    pts2d_cam = Prect.dot(pts3d_cam[:, idx])
    return pts3d[:, idx], pts2d_cam/pts2d_cam[2, :], idx


def lidar_to_camera_point(points, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T

    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.MATRIX_T_VELO_2_CAM)

    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.MATRIX_R_RECT_0)

    points = np.matmul(T_VELO_2_CAM, points)
    points = np.matmul(R_RECT_0, points).T
    points = points[:, 0:3]
    return points.reshape(-1, 3)

def lidar_to_bird_view(x, y):

    a = ((x - cfg.X_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)  # int
    b = ((y - cfg.Y_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)  # int

    return a, b

def box3d_to_label(result):
    ## waiting adding code
    return result


def label_to_gt_box3d(labels, cls='Car', T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    boxes3d = []
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    elif cls == 'Cyclist':
        acc_cls = ['Cyclist']
    else: # all
        acc_cls = []

    for label in labels:
        boxes_in_label = []
        for line in label:
            ret = line.split()
            if ret[0] in acc_cls or acc_cls == []:
                h, w, l, x_cam, y_cam, z_cam, ry_cam = [float(i) for i in ret[-7:]]
                x_l, y_l, z_l = camera_to_lidar(x_cam, y_cam, z_cam, T_VELO_2_CAM, R_RECT_0)
                rz_l = - ry_cam - np.pi / 2

                if rz_l <= - np.pi:
                    rz_l += 2 * np.pi #[-pi, pi]

                box3d = np.array([x_l, y_l, z_l, h, w, l, rz_l])
                boxes_in_label.append(box3d)
        boxes3d.append(np.array(boxes_in_label).reshape(-1, 7))#x,y,z,h,w,l,r

    return boxes3d


def delta_to_boxes3d(deltas, anchors):
    # Input:
    #   deltas: (N, w, l, 28)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 4, 7)

    # Output:
    #   boxes3d: (N, w*l*4, 7)
    # print('deltas', deltas.shape)# batchsize, 200, 176, 14
    # print('anchors', anchors.shape)# 200, 176, 4, 7
    anchors_reshaped = anchors.reshape(-1, 7)# 140800, 7 x y z h w l r
    deltas = deltas.reshape(deltas.shape[0], -1, 7)# batchsize, 140800, 7
    anchors_d = np.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)# d = sqrt(w^2 + l^2)
    boxes3d = np.zeros_like(deltas)

    boxes3d[..., [0, 1]] = deltas[..., [0, 1]] * anchors_d[:, np.newaxis] + anchors_reshaped[..., [0, 1]]# (¡÷x,¡÷y)*d+(x,y)
    boxes3d[..., [2]] = deltas[..., [2]] * cfg.ANCHOR_H + anchors_reshaped[..., [2]]# ¡÷z*h+z
    boxes3d[..., [3, 4, 5]] = np.exp(deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]#exp(¡÷h,¡÷w,¡÷l)*(h,w,l)
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]# ¡÷r+r

    return boxes3d


def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
                          np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d


# this just for visulize and testing
def lidar_box3d_to_camera_box(boxes3d, cal_projection=False, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)

    lidar_boxes3d_corner = center_to_corner_box3d(boxes3d)
    if type(P2) == type(None):
        P2 = np.array(cfg.MATRIX_P2)

    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d, T_VELO_2_CAM, R_RECT_0)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = int(np.min(points[:, 0]))
        maxx = int(np.max(points[:, 0]))
        miny = int(np.min(points[:, 1]))
        maxy = int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d


def lidar_to_bird_view_img(lidar, factor=1):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros(
        (cfg.INPUT_HEIGHT * factor, cfg.INPUT_WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.X_MIN < x < cfg.X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
            x, y = int((x - cfg.X_MIN) / cfg.VOXEL_X_SIZE *
                       factor), int((y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview


def cal_anchors():
    # Output:
    #   anchors: (w, l, 2, 7) x y z h w l r
    x = np.linspace(cfg.X_MIN, cfg.X_MAX-0.4, cfg.FEATURE_WIDTH)# 0 70.4-0.4 176
    y = np.linspace(cfg.Y_MIN, cfg.Y_MAX-0.4, cfg.FEATURE_HEIGHT)# -40 40-0.4 200

    cx, cy = np.meshgrid(x, y)# 200 176
    # all is (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 4)# 200 176 4
    cy = np.tile(cy[..., np.newaxis], 4)# 200 176 4
    cz = np.ones_like(cx) * cfg.ANCHOR_Z
    w = np.ones_like(cx) * cfg.ANCHOR_W
    l = np.ones_like(cx) * cfg.ANCHOR_L
    h = np.ones_like(cx) * cfg.ANCHOR_H
    r = np.ones_like(cx)
    r[..., 0] = 0
    r[..., 1] = np.pi
    r[..., 2] = np.pi / 2
    r[..., 3] = 3 * np.pi / 2

    # 7*(w,l,4) -> (w, l, 4, 7)
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)# 200 176 4 7
    return anchors


def filter_anchors(density_map, anchors):
    # Input:
    #   density_map: (w, l)# 800 704
    #   anchors: (N)# 200*176*4 7 x y z h w l r
    anchors_filtered = []
    num = anchors.shape[0]

    #plt.imshow(density_map)
    #plt.show()
    for anchor_id in range(0, num, 4):
        # 0 and pi
        x1 = anchors[anchor_id, 0] - anchors[anchor_id, 5] / 2 # l along x axis
        x2 = anchors[anchor_id, 0] + anchors[anchor_id, 5] / 2

        y1 = anchors[anchor_id, 1] - anchors[anchor_id, 4] / 2 # w along y axis
        y2 = anchors[anchor_id, 1] + anchors[anchor_id, 4] / 2

        x1_q = ((x1 - cfg.X_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)
        y1_q = ((y1 - cfg.Y_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)
        x2_q = ((x2 - cfg.X_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)
        y2_q = ((y2 - cfg.Y_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)

        x1_q, x2_q = np.clip(np.array([x1_q, x2_q]), 0, 704)
        y1_q, y2_q = np.clip(np.array([y1_q, y2_q]), 0, 800)

        if density_map[y1_q:y2_q, x1_q:x2_q].any():
            anchors_filtered.append(anchors[anchor_id])
            anchors_filtered.append(anchors[anchor_id + 1])

        # pi/2 and 3*pi/2
        x1 = anchors[anchor_id + 2, 0] - anchors[anchor_id + 2, 4] / 2  # w along x axis
        x2 = anchors[anchor_id + 2, 0] + anchors[anchor_id + 2, 4] / 2

        y1 = anchors[anchor_id + 2, 1] - anchors[anchor_id + 2, 5] / 2  # l along y axis
        y2 = anchors[anchor_id + 2, 1] + anchors[anchor_id + 2, 5] / 2
        x1_q = ((x1 - cfg.X_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)
        y1_q = ((y1 - cfg.Y_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)
        x2_q = ((x2 - cfg.X_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)
        y2_q = ((y2 - cfg.Y_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)

        x1_q, x2_q = np.clip(np.array([x1_q, x2_q]), 0, 704)
        y1_q, y2_q = np.clip(np.array([y1_q, y2_q]), 0, 800)

        if density_map[y1_q:y2_q, x1_q:x2_q].any():
            anchors_filtered.append(anchors[anchor_id + 2])
            anchors_filtered.append(anchors[anchor_id + 3])

    ret = np.array(anchors_filtered)
    return ret


def cal_rpn_target(labels, density_maps, feature_map_shape, anchors, cls='Car'):
    # Input:
    #   labels: (N, N')
    #   feature_map_shape: (w/2, l/2)# 400 352
    #   anchors: (w/4, l/4, 4, 7) # 200 176 4 7
    # Output:
    #   pos_equal_one (N, w/2, l/2, 2)
    #   neg_equal_one (N, w/2, l/2, 2)
    #   targets (N, w/2, l/2, 14)
    # attention: cal IoU on birdview

    batch_size = labels.shape[0]
    batch_gt_boxes3d = label_to_gt_box3d(labels, cls=cls)
    # defined in eq(1) in 2.2
    anchors_all = anchors.reshape(-1, 7)# 200*176*4, 5

    pos_equal_one = np.zeros((batch_size, *feature_map_shape, 4))# batchsize, 400, 352, 4
    neg_equal_one = np.zeros((batch_size, *feature_map_shape, 4))# batchsize, 400, 352, 4
    targets = np.zeros((batch_size, *feature_map_shape, 28))# batchsize, 400, 352, 28

    for batch_id in range(batch_size):
        anchors_reshaped = filter_anchors(density_maps[batch_id], anchors_all)
        anchors_d = np.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)  # d=sqrt(w^2+l^2)
        anchors_reshaped_2d =anchors_reshaped[:, [0, 1, 4, 5, 6]]

        batch_gt_boxes2d = batch_gt_boxes3d[batch_id][:, [0, 1, 4, 5, 6]]# x,y,w,l,r

        iou = box2d_rotate_iou(anchors_reshaped_2d, batch_gt_boxes2d)

        # find anchor with highest iou(iou should also > 0)
        id_highest = np.argmax(iou.T, axis=1)
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > cfg.RPN_POS_IOU)

        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < cfg.RPN_NEG_IOU,
                                 axis=1) == iou.shape[1])[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 1

        # ATTENTION: index_z should be np.array
        targets[batch_id, index_x, index_y, np.array(index_z) * 7] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 0] - anchors_reshaped[id_pos, 0]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 1] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 1] - anchors_reshaped[id_pos, 1]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 2] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 2] - anchors_reshaped[id_pos, 2]) / cfg.ANCHOR_H
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 3] / anchors_reshaped[id_pos, 3])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 4] / anchors_reshaped[id_pos, 4])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 5] / anchors_reshaped[id_pos, 5])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 6] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 6] - anchors_reshaped[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, 2))
        neg_equal_one[batch_id, index_x, index_y, index_z] = 1
        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, 2))
        neg_equal_one[batch_id, index_x, index_y, index_z] = 0

    return pos_equal_one, neg_equal_one, targets
