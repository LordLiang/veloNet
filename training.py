import os
import argparse
import time
import tensorflow as tf

from config import cfg
from core.model import RPN3D
from utils.dataset_loader import iterate_data, sample_test_data


def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=160,
                        help='max epoch')
    parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                        help='set log tag')
    parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=2,
                        help='set batch size')
    parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                        help='set learning rate')
    parser.add_argument('--output-path', type=str, nargs='?',
                        default='./predictions', help='results output dir')
    parser.add_argument('-v', '--vis', type=bool, nargs='?', default=False,
                        help='set the flag to True if dumping visualizations')
    args = parser.parse_args()
    return args

def main(_):
    args = parse_args()
    train_dir = os.path.join(cfg.DATA_DIR, 'training')
    val_dir = os.path.join(cfg.DATA_DIR, 'validation')
    log_dir = os.path.join('./log', args.tag)
    save_model_dir = os.path.join('./save_model', args.tag)
    batch_size = args.single_batch_size * cfg.GPU_USE_COUNT

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

    with tf.Graph().as_default():
        # for gpu setting
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                    visible_device_list=cfg.GPU_AVAILABLE,
                                    allow_growth=True)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={
                "GPU": cfg.GPU_USE_COUNT,
            },
            allow_soft_placement=True,
        )

        start_epoch = 0
        global_counter = 0

        with tf.Session(config=config) as sess:
            model = RPN3D(
                cls = cfg.DETECT_OBJ,
                single_batch_size=args.single_batch_size,
                learning_rate=args.lr,
                max_gradient_norm=5.0,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )

            if tf.train.get_checkpoint_state(save_model_dir):
                print("Reading model parameters from %s" % save_model_dir)
                model.saver.restore(
                    sess, tf.train.latest_checkpoint(save_model_dir))
                start_epoch = model.epoch.eval() + 1
                global_counter = model.global_step.eval() + 1
            else:
                print("Created model with fresh parameters.")
                tf.global_variables_initializer().run()

            # for summary setting
            summary_train_interval = 5
            summary_val_interval = 10
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            # training
            for epoch in range(start_epoch, args.max_epoch):
                counter = 0
                batch_time = time.time()
                for batch in iterate_data(train_dir, shuffle=True, batch_size=batch_size, aug=True, is_testset=False):
                    counter += 1
                    global_counter += 1

                    is_summary = (counter % summary_train_interval == 0)

                    start_time = time.time()
                    ret = model.train_step(sess, batch, train=True, summary=is_summary)
                    forward_time = time.time() - start_time
                    batch_time = time.time() - batch_time

                    print(
                        'train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f}'.format(
                            counter, epoch, args.max_epoch, ret[0], ret[1], ret[2], ret[3], ret[4], forward_time,
                            batch_time))
                    with open('log/train.txt', 'a') as f:
                        f.write(
                            'train: {} @ epoch:{}/{} loss: {:.4f} reg_loss: {:.4f} cls_loss: {:.4f} cls_pos_loss: {:.4f} cls_neg_loss: {:.4f} forward time: {:.4f} batch time: {:.4f} \n'.format(
                                counter, epoch, args.max_epoch, ret[0], ret[1], ret[2], ret[3], ret[4], forward_time,
                                batch_time))

                    if counter % summary_train_interval == 0:
                        print("summary_train_interval now")
                        summary_writer.add_summary(ret[-1], global_counter)

                    if counter % summary_val_interval == 0:
                        print("summary_val_interval now")
                        val_data = sample_test_data(val_dir, batch_size=batch_size)
                        ret = model.validate_step(sess, val_data, summary=True)
                        summary_writer.add_summary(ret[-1], global_counter)


                        ret = model.predict_step(sess, val_data, summary=True)
                        summary_writer.add_summary(ret[-1], global_counter)

                        print("prediction skipped due to error")

                    batch_time = time.time()


                sess.run(model.epoch_add_op)

                model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)

                # dump test data every 10 epochs
                # waiting for adding


            print('train done. total epoch:{} iter:{}'.format(
                epoch, model.global_step.eval()))

            # finally save model
            model.saver.save(sess, os.path.join(
                save_model_dir, 'checkpoint'), global_step=model.global_step)



if __name__ == '__main__':
    tf.app.run(main)
