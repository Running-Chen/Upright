'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime

import numpy as np
import tensorflow as tf
import socket
import importlib
import zipfile
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
# print(BASE_DIR)
# print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import dataset


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='uomodel',
                    help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument(
    '--save_folder', default='../../output/uoNet', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048,
                    help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=61,
                    help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=6,
                    help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000,
                    help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

SAVE_DIR = os.path.join(FLAGS.save_folder, '%s_%s' % (
    FLAGS.model, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

LOG_FOUT = open(os.path.join(SAVE_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

# bkp of code
code_folder = os.path.abspath(os.path.dirname(__file__))
zip_name = os.path.join(SAVE_DIR) + "/code.zip"
filelist = []
for root, dirs, files in os.walk(code_folder):
    for name in files:
        filelist.append(os.path.join(root, name))
zip_code = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
for tar in filelist:
    arcname = tar[len(code_folder):]
    zip_code.write(tar, arcname)
zip_code.close()

folder_ckpt = os.path.join(SAVE_DIR, 'ckpts')
if not os.path.exists(folder_ckpt):
    os.makedirs(folder_ckpt)

folder_summary = os.path.join(SAVE_DIR, 'summary')
if not os.path.exists(folder_summary):
    os.makedirs(folder_summary)

print(os.getcwd())
DATA_PATH = os.path.join(ROOT_DIR, '../../data')
TRAIN_DATASET = dataset.UoDataset(
    data_path=DATA_PATH, num_point=NUM_POINT, split='train', batch_size=BATCH_SIZE)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    # CLIP THE LEARNING RATE!
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch*BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, reg_pl = MODEL.placeholder_inputs(
                BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss ---")
            # Get model and loss
            pred_reg = MODEL.get_model(
                pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            
            loss_reg = MODEL.get_loss(pred_reg, reg_pl)
            
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            # tf.summary.scalar('total_loss', total_loss)
            # for l in losses + [total_loss]:
            #     tf.summary.scalar(l.op.name, l)

            # correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(reg_pl))
            # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            # tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            os.path.join(folder_summary, 'train'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'reg_pl': reg_pl,
               'is_training_pl': is_training_pl,
               'pred_reg': pred_reg,
               'loss_reg': loss_reg,

               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)

            # Save the variables to disk.
            if epoch % 30 == 0:
                save_path = saver.save(sess, os.path.join(
                    folder_ckpt, "model.ckpt"), global_step=epoch)
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)//BATCH_SIZE

    loss_sum_reg = 0
    batch_idx = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1)*BATCH_SIZE
        batch_pc, batch_uo = TRAIN_DATASET.get_batch(
            train_idxs, start_idx, end_idx)
        
        feed_dict = {ops['pointclouds_pl']: batch_pc,
                     ops['reg_pl']: batch_uo,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_reg_val, pred_reg_val,reg_pl = sess.run([ops['merged'], ops['step'],
                                                                 ops['train_op'], ops['loss_reg'], ops['pred_reg'], ops['reg_pl']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        # pred_val = np.argmax(pred_val, 1)
        # correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        # total_correct += correct
        # total_seen += bsize
        # loss_sum += loss_val
        
        loss_sum_reg += loss_reg_val

        if batch_idx % 1000 == 0 and batch_idx != 0:
            log_string(' -- %03d / %03d --' % (batch_idx, num_batches))
            log_string('mean reg loss: %f' % (loss_sum_reg / 1000))
            loss_sum_reg = 0
            log_string('pred_reg %s' % str(pred_reg_val))
            log_string('reg_pl %s' % reg_pl)
            

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
