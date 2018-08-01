import os
import glob
import numpy as np
import math
import multiprocessing

from config import cfg

CAM =2

class Dataset:
    def __init__(self, data_tag, f_bev, f_label, data_dir, aug, is_testset):
        self.data_tag = data_tag
        self.f_bev = f_bev
        self.f_label = f_label
        self.data_dir = data_dir
        self.aug = aug
        self.is_testset = is_testset


    def __call__(self, load_index):
        #if self.aug:
        tag = self.data_tag[load_index]
        bev_map = np.fromfile(self.f_bev[load_index], dtype=np.float32).reshape((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, cfg.CHANNEL_SIZE))

        if not self.is_testset:
            f_read = open(self.f_label[load_index], 'r')
            labels = [line for line in f_read.read().splitlines()]
            f_read.close()
        else:
            labels = ['']
        ret = [tag, bev_map, labels]
        return ret


# global pool
TRAIN_POOL = multiprocessing.Pool(8)
VAL_POOL = multiprocessing.Pool(4)


def iterate_data(data_dir, shuffle=False, batch_size=1, aug=False, is_testset=False):
    f_bev = glob.glob(os.path.join(data_dir, 'bev', '*.bin'))
    f_label = glob.glob(os.path.join(data_dir, 'label', '*.txt'))
    f_bev.sort()
    f_label.sort()

    data_tag = [name.split('/')[-1].split('.')[-2] for name in f_bev]
    nums = len(f_bev)
    indices = list(range(nums))
    if shuffle:
        np.random.shuffle(indices)

    num_batches = int(math.floor(nums / float(batch_size)))

    dataset = Dataset(data_tag, f_bev, f_label, data_dir, aug, is_testset)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        excerpt = indices[start_idx:start_idx + batch_size]

        rets = TRAIN_POOL.map(dataset, excerpt)

        tag = [ret[0] for ret in rets ]
        bev_map = [ret[1] for ret in rets ]
        labels = [ret[2] for ret in rets ]

        ret = (np.array(tag), np.array(bev_map), np.array(labels))
        yield ret

def sample_test_data(data_dir, batch_size=1):
    f_bev = glob.glob(os.path.join(data_dir, 'bev', '*.bin'))
    f_label = glob.glob(os.path.join(data_dir, 'label', '*.txt'))

    f_bev.sort()
    f_label.sort()

    data_tag = [name.split('/')[-1].split('.')[-2] for name in f_bev]

    assert (len(data_tag) == len(f_bev) == len(f_label)), "dataset folder is not correct"

    nums = len(f_bev)

    indices = list(range(nums))
    np.random.shuffle(indices)

    excerpt = indices[0:batch_size]

    dataset = Dataset(data_tag, f_bev, f_label, data_dir, False, False)

    rets = VAL_POOL.map(dataset, excerpt)

    tag = [ret[0] for ret in rets]
    bev_map = [ret[1] for ret in rets]
    labels = [ret[2] for ret in rets]

    ret = (np.array(tag), np.array(bev_map), np.array(labels))

    return ret

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

if __name__ == '__main__':
    pass

