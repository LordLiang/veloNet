import tensorflow as tf
import os
from config import cfg
from core.featureNet import FeatureNet
from core.rpn import RPN
from utils.utils import *
from utils.dataset_loader import *
from utils.op import average_gradients
from utils.rotate_iou import box2d_rotate_nms
from utils.draw import *
from utils.colorize import colorize
from scipy.misc import imread

class RPN3D(object):

    def __init__(self,
                 cls='Car',
                 single_batch_size=1,
                 learning_rate=0.001,
                 max_gradient_norm=5.0,
                 avail_gpus=['0']):
        # hyper parameters and status
        self.cls = cls
        self.single_batch_size = single_batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.avail_gpus = avail_gpus

        boundaries = [80, 120]
        values = [self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01]
        lr = tf.train.piecewise_constant(self.epoch, boundaries, values)

        # build graph
        # input placeholders
        self.is_train = tf.placeholder(tf.bool, name='phase')

        self.bev_maps = []
        self.targets = []
        self.pos_equal_one = []
        self.pos_equal_one_sum = []
        self.pos_equal_one_for_reg = []
        self.neg_equal_one = []
        self.neg_equal_one_sum = []

        self.delta_output = []
        self.prob_output = []
        self.opt = tf.train.AdamOptimizer(lr)
        self.gradient_norm = []
        self.tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            for idx, dev in enumerate(self.avail_gpus):
                with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
                    # graph
                    feature = FeatureNet(
                        training=self.is_train, batch_size=self.single_batch_size)
                    rpn = RPN(
                        input=feature.outputs, training=self.is_train)
                    tf.get_variable_scope().reuse_variables()
                    # input
                    self.bev_maps.append(feature.input)
                    self.targets.append(rpn.targets)
                    self.pos_equal_one.append(rpn.pos_equal_one)
                    self.pos_equal_one_sum.append(rpn.pos_equal_one_sum)
                    self.pos_equal_one_for_reg.append(rpn.pos_equal_one_for_reg)
                    self.neg_equal_one.append(rpn.neg_equal_one)
                    self.neg_equal_one_sum.append(rpn.neg_equal_one_sum)
                    # output
                    delta_output = rpn.delta_output
                    prob_output = rpn.prob_output
                    # loss and grad
                    if idx == 0:
                        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    self.loss = rpn.loss
                    self.reg_loss = rpn.reg_loss
                    self.cls_loss = rpn.cls_loss
                    self.cls_pos_loss = rpn.cls_pos_loss_rec
                    self.cls_neg_loss = rpn.cls_neg_loss_rec
                    self.params = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, self.params)
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

                    self.delta_output.append(delta_output)
                    self.prob_output.append(prob_output)
                    self.tower_grads.append(clipped_gradients)
                    self.gradient_norm.append(gradient_norm)
                    self.rpn_output_shape = rpn.output_shape

        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # loss and optimizer
        # self.xxxloss is only the loss for the lowest tower
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.grads = average_gradients(self.tower_grads)
            self.update = [self.opt.apply_gradients(
                zip(self.grads, self.params), global_step=self.global_step)]
            self.gradient_norm = tf.group(*self.gradient_norm)

        self.update.extend(self.extra_update_ops)
        self.update = tf.group(*self.update)

        self.delta_output = tf.concat(self.delta_output, axis=0)
        self.prob_output = tf.concat(self.prob_output, axis=0)

        self.anchors = cal_anchors()
        # for predict and image summary
        self.rgb = tf.placeholder(
            tf.uint8, [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])
        self.bv = tf.placeholder(tf.uint8, [
            None, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 3])
        self.bv_heatmap = tf.placeholder(tf.uint8, [
            None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 3])

        self.boxes2d = tf.placeholder(tf.float32, [None, 5])# x y w l r
        self.boxes2d_scores = tf.placeholder(tf.float32, [None])

        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            # summary and saver
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                        max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

            self.train_summary = tf.summary.merge([
                tf.summary.scalar('train/loss', self.loss),
                tf.summary.scalar('train/reg_loss', self.reg_loss),
                tf.summary.scalar('train/cls_loss', self.cls_loss),
                tf.summary.scalar('train/cls_pos_loss', self.cls_pos_loss),
                tf.summary.scalar('train/cls_neg_loss', self.cls_neg_loss),
                *[tf.summary.histogram(each.name, each) for each in self.vars + self.params]
            ])

            self.validate_summary = tf.summary.merge([
                tf.summary.scalar('validate/loss', self.loss),
                tf.summary.scalar('validate/reg_loss', self.reg_loss),
                tf.summary.scalar('validate/cls_loss', self.cls_loss),
                tf.summary.scalar('validate/cls_pos_loss', self.cls_pos_loss),
                tf.summary.scalar('validate/cls_neg_loss', self.cls_neg_loss)
            ])

        self.predict_summary = tf.summary.merge([
            tf.summary.image('predict/bird_view_lidar', self.bv),
            tf.summary.image('predict/bird_view_heatmap', self.bv_heatmap),
            tf.summary.image('predict/front_view_rgb', self.rgb),
        ])

    def train_step(self, session, data, train=False, summary=False):

        tags = data[0]
        bev_maps = data[1]
        labels = data[2]

        density_maps = bev_maps[..., 5]
        print('train', tags)
        pos_equal_one, neg_equal_one, targets = cal_rpn_target(
            labels, density_maps, self.rpn_output_shape, self.anchors, cls=cfg.DETECT_OBJ)

        temp1 = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
        temp2 = np.concatenate([np.tile(pos_equal_one[..., [1]], 7), np.tile(pos_equal_one[..., [2]], 7)], axis=-1)
        pos_equal_one_for_reg = np.concatenate([temp1, temp2], axis=-1)

        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

        input_feed = {}
        input_feed[self.is_train] = True
        for idx in range(len(self.avail_gpus)):
            input_feed[self.bev_maps[idx]] = bev_maps[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.targets[idx]] = targets[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[idx]] = pos_equal_one[
                                                 idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[idx]] = pos_equal_one_sum[
                                                      idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[idx]] = pos_equal_one_for_reg[
                                                          idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[idx]] = neg_equal_one[
                                                  idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[idx]] = neg_equal_one_sum[
                                                      idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
        if train:
            output_feed = [self.loss, self.reg_loss,
                           self.cls_loss, self.cls_pos_loss, self.cls_neg_loss, self.gradient_norm, self.update]
        else:
            output_feed = [self.loss, self.reg_loss, self.cls_loss, self.cls_pos_loss, self.cls_neg_loss]
        if summary:
            output_feed.append(self.train_summary)

        return session.run(output_feed, feed_dict=input_feed)



    def validate_step(self, session, data, summary=False):
        tags = data[0]
        bev_maps = data[1]
        labels = data[2]
        density_maps = bev_maps[..., 5]

        print('valid', tags)
        pos_equal_one, neg_equal_one, targets = cal_rpn_target(
            labels, density_maps, self.rpn_output_shape, self.anchors, cls=cfg.DETECT_OBJ)

        temp1 = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
        temp2 = np.concatenate([np.tile(pos_equal_one[..., [1]], 7), np.tile(pos_equal_one[..., [2]], 7)], axis=-1)
        pos_equal_one_for_reg = np.concatenate([temp1, temp2], axis=-1)

        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

        input_feed = {}
        input_feed[self.is_train] = False
        for idx in range(len(self.avail_gpus)):
            input_feed[self.bev_maps[idx]] = bev_maps[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.targets[idx]] = targets[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[idx]] = pos_equal_one[
                                                  idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[idx]] = pos_equal_one_sum[
                                                      idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[idx]] = pos_equal_one_for_reg[idx * self.single_batch_size:(
                                                                                                             idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[idx]] = neg_equal_one[
                                                  idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[idx]] = neg_equal_one_sum[
                                                      idx * self.single_batch_size:(idx + 1) * self.single_batch_size]

        output_feed = [self.loss, self.reg_loss, self.cls_loss]
        if summary:
            output_feed.append(self.validate_summary)
        return session.run(output_feed, input_feed)


    def predict_step(self, session, data, summary=False, vis=False):
        tags = data[0]
        bev_maps = data[1]
        labels = data[2]
        batch_size = len(tags)
        if summary or vis:
            batch_gt_boxes3d = label_to_gt_box3d(labels, cls=self.cls)

        print('predict', tags)
        input_feed = {}
        input_feed[self.is_train] = False
        for idx in range(len(self.avail_gpus)):
            input_feed[self.bev_maps[idx]] = bev_maps[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]

        output_feed = [self.prob_output, self.delta_output]
        probs, deltas = session.run(output_feed, input_feed)
        print('probs', probs.shape)
        print('deltas', deltas.shape)

        batch_boxes3d = delta_to_boxes3d(deltas, self.anchors)
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        batch_probs = probs.reshape((len(self.avail_gpus) * self.single_batch_size, -1))

        # NMS
        print("NMS...")
        ret_box3d = []
        ret_score = []

        for batch_id in range(batch_size):
            # remove box with low score
            ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]

            ind = box2d_rotate_nms(tmp_boxes2d,
                                   tmp_scores,
                                   max_output_size=cfg.RPN_NMS_POST_TOPK,
                                   iou_threshold=cfg.RPN_NMS_THRESH)

            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]

            inf = float("inf")

            for i in range(len(ind)-1, -1, -1):
                if 0 in tmp_boxes3d[i, 3:6] or inf in tmp_boxes3d[i, 3:6]:
                    tmp_boxes3d = np.delete(tmp_boxes3d, i, 0)
                    tmp_scores = np.delete(tmp_scores, i, 0)

            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))
        if summary:
            # only summry 1 in a batch
            cur_tag = tags[0]
            cur_img = imread(os.path.join(cfg.IMG_DIR, cur_tag + '.png'))
            P, Tr, R = load_calib(os.path.join(cfg.CALIB_DIR, cur_tag + '.txt'))

            front_image = draw_box3d_on_image(cur_img, ret_box3d[0], ret_score[0],
                                                    batch_gt_boxes3d[0], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)

            bird_view = bev_maps[..., 7:10]


            bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0],
                                                     batch_gt_boxes3d[0], factor=cfg.BV_LOG_FACTOR, P2=P,
                                                     T_VELO_2_CAM=Tr, R_RECT_0=R)

            heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)

            ret_summary = session.run(self.predict_summary, {
                self.rgb: front_image[np.newaxis, ...],
                self.bv: bird_view[np.newaxis, ...],
                self.bv_heatmap: heatmap[np.newaxis, ...]
            })

            return tags, ret_box3d_score, ret_summary

        if vis:
            front_images, bird_views, heatmaps = [], [], []
            for i in range(batch_size):
                cur_tag = tags[i]
                P, Tr, R = load_calib(os.path.join(cfg.CALIB_DIR, cur_tag + '.txt'))

                bird_view = bev_maps[..., 7:10]

                bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[i], ret_score[i],
                                                         batch_gt_boxes3d[i], factor=cfg.BV_LOG_FACTOR, P2=P,
                                                         T_VELO_2_CAM=Tr, R_RECT_0=R)

                heatmap = colorize(probs[i, ...], cfg.BV_LOG_FACTOR)

                bird_views.append(bird_view)
                heatmaps.append(heatmap)

            return tags, ret_box3d_score, bird_views, heatmaps

        return tags, ret_box3d_score





