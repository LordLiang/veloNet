from utils.utils import *
import cv2



def draw_box3d_on_image(img, boxes3d, scores, gt_boxes3d=np.array([]),
                        color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1,
                        P2=None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM,
                                            R_RECT_0=R_RECT_0)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM,
                                               R_RECT_0=R_RECT_0)

    # draw projections
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    # draw gt projections
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, gt_boxes3d=np.array([]),
                                 color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1, factor=1,
                                 P2=None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    corner_boxes3d = center_to_corner_box3d(boxes3d, coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM,
                                               R_RECT_0=R_RECT_0)
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2])
        x1, y1 = lidar_to_bird_view(*box[1, 0:2])
        x2, y2 = lidar_to_bird_view(*box[2, 0:2])
        x3, y3 = lidar_to_bird_view(*box[3, 0:2])

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)

    # draw detections
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2])
        x1, y1 = lidar_to_bird_view(*box[1, 0:2])
        x2, y2 = lidar_to_bird_view(*box[2, 0:2])
        x3, y3 = lidar_to_bird_view(*box[3, 0:2])

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)