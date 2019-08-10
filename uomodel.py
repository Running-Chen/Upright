from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tf_util
import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(
        tf.float32, shape=(batch_size, num_point, 3))
    uo_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    return pointclouds_pl, uo_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx3 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    l0_xyz = point_cloud
    l0_points = None

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    ''' shape=(batch_size, 1024, 128) '''
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=64, mlp=[64, 64, 128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer1')
    ''' shape=(batch_size, 512, 256) '''
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.2, nsample=64, mlp=[128, 128, 256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer2')
    ''' shape=(batch_size, 128, 512) '''
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4, nsample=64, mlp=[256, 256, 512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sa_layer3')
    ''' shape=(batch_size, 1, 1024) '''
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[512, 512, 1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='sa_layer4')

    # Fully connected layers
    net = tf.reshape(l4_points, [batch_size, -1])
    net = tf_util.fully_connected(
        net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5,
                          is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(
        net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5,
                          is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 3, activation_fn=None, scope='fc3')
    
    return net

def get_loss(pred_uo, gt_uo):
    """ pred: BX3,
        label: BX3"""

    # loss = tf.sqrt(tf.reduce_sum(tf.square(pred-label),1))
    
    pred_norm = tf.sqrt(tf.reduce_sum(tf.square(pred_uo),axis=1))
    gt_norm = tf.sqrt(tf.reduce_sum(tf.square(gt_uo),axis=1))
    pred_gt = tf.reduce_sum(tf.multiply(pred_uo,gt_uo),axis=1)
    cosin = pred_gt/(pred_norm * gt_norm)
    angle_err = tf.acos(cosin)
    
    loss_reg = tf.reduce_mean(angle_err)
    
        # sess.run(loss_reg)
    #     for(pred,gt) in (pred_uo,gt_uo):

    #         cos = np.dot(pred_uo, gt_uo) / \
    #                     (eps + np.linalg.norm(pred_uo)*np.linalg.norm(gt_uo))
    #         angle_err = np.arccos(cos)  # [0,PI]
    #         m_err = m_err+angle_err
    # loss_reg = m_err / num


    tf.summary.scalar('loss_reg', loss_reg)
    tf.add_to_collection('losses', loss_reg) #put the loss into the set

    return loss_reg

if __name__ =='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((6,2048,3))
        output = get_model(inputs, tf.constant(True))
        print(output)
