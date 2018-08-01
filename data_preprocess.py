from utils.converter import *
from config import cfg
import math
import matplotlib.pyplot as plt
from scipy.misc import imread

CAM = 2
# for preprocessing dataset dir
TRAIN_IMG_ROOT = '/adata/zhoujie/KITTI/object/training/image_2/'
TRAIN_PC_ROOT = '/adata/zhoujie/KITTI/object/training/velodyne/'
TRAIN_CALIB_ROOT = '/adata/zhoujie/KITTI/object/training/calib/'
TRAIN_LABEL_ROOT = '/adata/zhoujie/KITTI/object/training/label_2/'
TRAIN_BEV_ROOT = '/adata/zhoujie/KITTI/object/training/bev/'
TRAIN_NEW_LABEL_ROOT = '/adata/zhoujie/KITTI/object/training/new_label_2/'

TEST_IMG_ROOT = '/adata/zhoujie/KITTI/object/testing/image_2/'
TEST_PC_ROOT = '/adata/zhoujie/KITTI/object/testing/velodyne/'
TEST_CALIB_ROOT = '/adata/zhoujie/KITTI/object/testing/calib/'
TEST_LABEL_ROOT = '/adata/zhoujie/KITTI/object/testing/label_2/'
TEST_BEV_ROOT = '/adata/zhoujie/KITTI/object/testing/bev/'
TEST_NEW_LABEL_ROOT = '/adata/zhoujie/KITTI/object/testing/new_label_2/'

def load_calib(calib_dir):
    lines = open(calib_dir).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]
    #P2 3*4
    P = np.array(lines[CAM]).reshape(3, 4)
    #Tr_velo_to_cam 3*4 --> 4*4
    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1,4)], 0)
    #R0_rect 3*3
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)

    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points

def filter_center_car(points):
    idx = np.where(np.logical_or(np.abs(points[:, 0]) > 4.7/2, np.abs(points[:, 1]) > 2.1/2))
    points = points[idx]
    return points

def remove_out_of_range(points):
    idx = np.where((points[:, 0] >= cfg.X_MIN) & (points[:, 0] <= cfg.X_MAX))
    points = points[idx]
    idx = np.where((points[:, 1] >= cfg.Y_MIN) & (points[:, 1] <= cfg.Y_MAX))
    points = points[idx]
    idx = np.where((points[:, 2] >= cfg.Z_MIN) & (points[:, 2] <= cfg.Z_MAX))
    points = points[idx]
    return points

def align_img_and_pc(img_dir, pc_dir, calib_dir):
    img = imread(img_dir)
    pts = load_velodyne_points(pc_dir)
    P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir)

    pts3d = pts.copy()
    '''Replaces the reflectance value by 1, and tranposes the array, so
            points can be directly multiplied by the camera projection matrix'''
    pts3d[:, 3] = 1
    pts3d = pts3d.transpose()
    pts3d, pts2d_normed, idx = project_velo_points_in_img(pts3d, Tr_velo_to_cam, R_cam_to_rect, P)

    reflectances = pts[:, 3]
    reflectances = reflectances[idx]

    assert reflectances.shape[0] == pts3d.shape[1] == pts2d_normed.shape[1]

    rows, cols = img.shape[:2]

    points = []
    for i in range(pts2d_normed.shape[1]):
        c = int(np.round(pts2d_normed[0, i]))
        r = int(np.round(pts2d_normed[1, i]))
        if c < cols and r < rows and r > 0 and c > 0:
            color = img[r, c, :]
            point = [pts3d[0, i], pts3d[1, i], pts3d[2, i], reflectances[i], color[0], color[1], color[2]]
            points.append(point)

    points = np.array(points)
    return points

def generate_bev(points):

    p_x = points[:, 0]
    p_y = points[:, 1]
    p_z = points[:, 2]
    p_r = points[:, 3]
    p_color = points[:, 4:]/255

    p_x_quantized = ((p_x - cfg.X_MIN) // cfg.GRID_RESOLUTION).astype(np.int32) # int
    p_y_quantized = ((p_y - cfg.Y_MIN) // cfg.GRID_RESOLUTION).astype(np.int32) # int
    p_z_quantized = (p_z - cfg.Z_MIN) / cfg.GRID_RESOLUTION # float

    quantized = np.dstack((p_x_quantized, p_y_quantized, p_z_quantized, p_r, p_color[:, 0], p_color[:, 1], p_color[:, 2])).squeeze()

    X0, Xn = 0, math.ceil((cfg.X_MAX - cfg.X_MIN) / cfg.GRID_RESOLUTION)# 0 704
    Y0, Yn = 0, math.ceil((cfg.Y_MAX - cfg.Y_MIN) / cfg.GRID_RESOLUTION)# 0 800
    Z0, Zn = 0, math.ceil((cfg.Z_MAX - cfg.Z_MIN) / cfg.GRID_RESOLUTION)# 0 40

    width = Xn - X0 # 704
    height = Yn - Y0 # 800
    slice = math.ceil((Zn - Z0)/cfg.SLICE_SIZE) # 5
    channel = slice + 5 # 5 + 2 + 3

    bev = np.zeros(shape=(height, width, channel), dtype=np.float32)# 800*704*10

    for y in range(height):
        index_y = np.where(quantized[:, 1] == y)
        quantized_y = quantized[index_y]
        if len(quantized_y) == 0:
            continue

        for x in range(width):
            index_x = np.where(quantized_y[:, 0] == x)
            quantized_x_y = quantized_y[index_x]
            count = len(quantized_x_y)# num of points in grid(x,y)
            if count == 0:
                continue

            # density map
            density = min(1, np.log(count + 1) / math.log(64))
            bev[y, x, slice] = density

            # intensity map
            highest_point = np.argmax(quantized_x_y[:, 2])
            bev[y, x, slice + 1] = quantized_x_y[highest_point, 3]

            # color map
            bev[y, x, (slice + 2):(slice + 5)] = quantized_x_y[highest_point, 4:7]

            # height map per slice
            for s in range(slice):
                index_z = np.where((quantized_x_y[:, 2] >= s * cfg.SLICE_SIZE) & (quantized_x_y[:, 2] <= (s + 1) * cfg.SLICE_SIZE))
                quantized_x_y_z = quantized_x_y[index_z]
                if len(quantized_x_y_z) == 0:
                    continue
                max_height = max(0, np.max(quantized_x_y_z[:, 2]) - s * cfg.SLICE_SIZE)
                bev[y, x, s] = max_height

    return bev

def generate_label(LABEL_ROOT, NEW_LABEL_ROOT, num):
    for frame in range(0, num):
        label_dir = LABEL_ROOT + '%06d.txt' % frame
        f_read = open(label_dir, 'r')
        labels = [line for line in f_read.read().splitlines()]
        f_read.close()

        new_labels = []
        for label in labels:
            label_split = label.split(' ')
            type = label_split[0]
            if (type == 'Car') | (type == 'Pedestrian') | (type == 'Cyclist'):
                h, w, l, x, y, z, r = label_split[8:15]
                x, y, z = camera_to_lidar(float(x),float(y),float(z))

                h_q = float(h) / cfg.GRID_RESOLUTION
                w_q = float(w) / cfg.GRID_RESOLUTION
                l_q = float(l) / cfg.GRID_RESOLUTION
                x_q = ((x - cfg.X_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)
                y_q = ((y - cfg.Y_MIN) // cfg.GRID_RESOLUTION).astype(np.int32)
                z_q = (z - cfg.Z_MIN) / cfg.GRID_RESOLUTION
                rz = -float(r) - np.pi / 2
                new_label = type + ' %.2f %.2f %.2f %d %d %.2f %.2f\n' % (h_q, w_q, l_q, x_q, y_q, z_q, rz)
                new_labels.append(new_label)

        output_name = NEW_LABEL_ROOT + '%06d.txt' % frame
        f_write = open(output_name, 'w')
        f_write.writelines(new_labels)
        f_write.close()

def save_bev(IMG_ROOT, PC_ROOT, CALIB_ROOT, BEV_ROOT, num):
    for frame in range(0, num):
        img_dir = IMG_ROOT + '%06d.png' % frame
        pc_dir = PC_ROOT + '%06d.bin' % frame
        calib_dir = CALIB_ROOT + '%06d.txt' % frame

        points = align_img_and_pc(img_dir, pc_dir, calib_dir)
        points = remove_out_of_range(points)
        points = filter_center_car(points)
        bev_maps = generate_bev(points)
        #bev_rgb = bev_maps[:,:,7:10]
        #plt.imshow(bev_rgb)
        #plt.savefig('data/' + '%06d.png' % frame)
        #plt.show()

        output_name = BEV_ROOT + '%06d.bin' % frame
        bev_maps.astype('float32').tofile(output_name)

if __name__ == '__main__':

    #print('generate new labels for bev map...')
    #generate_label(TRAIN_LABEL_ROOT, TRAIN_NEW_LABEL_ROOT, 10)

    print('generate bev map from the raw point cloud...')
    save_bev(TRAIN_IMG_ROOT, TRAIN_PC_ROOT, TRAIN_CALIB_ROOT, TRAIN_BEV_ROOT, 7481)
    save_bev(TEST_IMG_ROOT, TEST_PC_ROOT, TEST_CALIB_ROOT, TEST_BEV_ROOT, 7518)


