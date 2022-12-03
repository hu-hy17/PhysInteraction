import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


def zero_padding(inputs, pad_1, pad_2):
  pad_mat = np.array([[0, 0], [pad_1, pad_2], [pad_1, pad_2], [0, 0]])
  return tf.pad(inputs, paddings=pad_mat)


def conv_bn(inputs, oc, ks, st, scope, training, rate=1):
  with tf.variable_scope(scope):
    if st == 1:
      layer = tf.layers.conv2d(
        inputs, oc, ks, strides=st, padding='SAME', use_bias=False,
        dilation_rate=rate,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
        kernel_initializer=tf.contrib.layers.xavier_initializer()
      )
    else:
      pad_total = ks - 1
      pad_1 = pad_total // 2
      pad_2 = pad_total - pad_1
      padded_inputs = zero_padding(inputs, pad_1, pad_2)
      layer = tf.layers.conv2d(
        padded_inputs, oc, ks, strides=st, padding='VALID', use_bias=False,
        dilation_rate=rate,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
        kernel_initializer=tf.contrib.layers.xavier_initializer()
      )
    layer = tf.layers.batch_normalization(layer, training=training)
  return layer


def conv_bn_relu(inputs, oc, ks, st, scope, training, rate=1):
  layer = conv_bn(inputs, oc, ks, st, scope, training, rate=rate)
  layer = tf.nn.relu(layer)
  return layer


def deconv_bn(inputs, oc, ks, st, scope, training):
  with tf.variable_scope(scope):
    layer = tf.layers.conv2d_transpose(
      inputs, oc, ks, strides=st, padding='SAME', use_bias=False,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
      kernel_initializer=tf.contrib.layers.xavier_initializer()
    )
    layer = tf.layers.batch_normalization(layer, training=training)
  return layer


def deconv_bn_relu(inputs, oc, ks, st, scope, training):
  layer = deconv_bn(inputs, oc, ks, st, scope, training)
  layer = tf.nn.relu(layer)
  return layer


def bottleneck(inputs, oc, st, scope, training, rate=1):
  with tf.variable_scope(scope):
    ic = inputs.get_shape().as_list()[-1]
    if ic == oc:
      if st == 1:
        shortcut = inputs
      else:
        shortcut = \
          tf.nn.max_pool2d(inputs, [1, st, st, 1], [1, st, st, 1], 'SAME')
    else:
      shortcut = conv_bn(inputs, oc, 1, st, 'shortcut', training)

    residual = conv_bn_relu(inputs, oc//4, 1, 1, 'conv1', training)
    residual = conv_bn_relu(residual, oc//4, 3, st, 'conv2', training, rate)
    residual = conv_bn(residual, oc, 1, 1, 'conv3', training)
    output = tf.nn.relu(shortcut + residual)

  return output


def resnet50(inputs, scope, training):
  with tf.variable_scope(scope):
    layers = []

    layers.append(inputs)
    layer = conv_bn_relu(inputs, 64, 7, 2, 'conv1', training)

    with tf.variable_scope('block1'):
      for unit in range(2):
        layer = bottleneck(layer, 256, 1, 'unit%d' % (unit+1), training)
      layers.append(layer)
      layer = bottleneck(layer, 256, 2, 'unit3', training)

    with tf.variable_scope('block2'):
      for unit in range(3):
        layer = bottleneck(layer, 512, 1, 'unit%d' % (unit+1), training, 2)
      layers.append(layer)
      layer = bottleneck(layer, 256, 2, 'unit4', training)

    with tf.variable_scope('block3'):
      for unit in range(6):
        layer = bottleneck(layer, 1024, 1, 'unit%d' % (unit+1), training, 4)

    layer = conv_bn_relu(layer, 256, 3, 1, 'squeeze', training)
    layers.append(layer)

  return layers


def net_2d(features, scope, training, n_out):
  with tf.variable_scope(scope):
    layer = conv_bn_relu(features, 256, 3, 1, 'project', training)
    with tf.variable_scope('prediction'):
      layer = tf.layers.conv2d(
        layer, n_out, 1, strides=1, padding='SAME',
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.1)
      )
  return layer


def net_z(features, scope, training, n_out):
  with tf.variable_scope(scope):
    layer = conv_bn_relu(features, 256, 3, 1, 'project', training)
    with tf.variable_scope('prediction'):
      layer = tf.layers.conv2d(
        layer, n_out, 1, strides=1, padding='SAME',
        activation=None,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.1)
      )
  return layer


def net_3d(features, scope, training, n_out):
  with tf.variable_scope(scope):
    layer = conv_bn_relu(features, 256, 3, 1, 'project', training)
    with tf.variable_scope('prediction'):
      layer = tf.layers.conv2d(
        layer, n_out * 3, 1, strides=1, padding='SAME',
        activation=None,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.3)
      )

  return layer


def net_seg(features, layers, scope, training):
  with tf.variable_scope(scope):
    curr_layer = features
    for i, skip_layer in enumerate(reversed(layers)):
      curr_layer = deconv_bn_relu(
        curr_layer, skip_layer.get_shape().as_list()[-1], 3, 2, 'deconv%d' % i,
        training
      )
    curr_layer = tf.concat([curr_layer, skip_layer], -1)
    layer = tf.layers.conv2d(
      curr_layer, 1, 1, strides=1, padding='SAME',
      activation=tf.nn.sigmoid,
      kernel_initializer=tf.initializers.truncated_normal(stddev=0.1)
    )

  return layer


def net_seg_aux(features, layers, scope, training):
  with tf.variable_scope(scope):
    curr_layer = features
    seg_layers = []
    for i, skip_layer in enumerate(reversed(layers)):
      curr_layer = deconv_bn_relu(
        curr_layer, skip_layer.get_shape().as_list()[-1], 3, 2, 'deconv%d' % i,
        training
      )
      curr_layer = tf.concat([curr_layer, skip_layer], -1)
      seg_layer = tf.layers.conv2d(
        curr_layer, 1, 1, strides=1, padding='SAME',
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.initializers.truncated_normal(stddev=0.1)
      )
      seg_layers.append(seg_layer)
      curr_layer = tf.concat([curr_layer, seg_layer], -1)

  return seg_layers


def baseline_net(inputs, scope, training):
  with tf.variable_scope(scope):
    feature_layers = resnet50(inputs, scope, training)
    hmap = net_2d(feature_layers[-1], 'hmap', training, 21)
    feature_bulk = tf.concat([feature_layers[-1], hmap], -1)
    dmap = net_3d(feature_bulk, 'dmap', training, 21)
    feature_bulk = tf.concat([feature_bulk, dmap], -1)
    lmap = net_3d(feature_bulk, 'lmap', training, 21)
    mask = net_seg(
      tf.concat([feature_layers[-1], hmap], -1), feature_layers[:-1],
      'mask', training
    )[..., -1]
    h, w = hmap.get_shape().as_list()[1:3]
    dmap = tf.reshape(dmap, [-1, h, w, 21, 3])
    lmap = tf.reshape(dmap, [-1, h, w, 21, 3])
  return hmap, dmap, lmap, mask


def aux_net(inputs, scope, training):
  with tf.variable_scope(scope):
    feature_layers = resnet50(inputs, scope, training)
    hmap = net_2d(feature_layers[-1], 'hmap', training, 21)
    feature_bulk = tf.concat([feature_layers[-1], hmap], -1)
    dmap = net_3d(feature_bulk, 'dmap', training, 21)
    feature_bulk = tf.concat([feature_bulk, dmap], -1)
    lmap = net_3d(feature_bulk, 'lmap', training, 21)
    masks = net_seg_aux(
      tf.concat([feature_layers[-1], hmap], -1), feature_layers[:-1],
      'mask', training
    )
    h, w = hmap.get_shape().as_list()[1:3]
    dmap = tf.reshape(dmap, [-1, h, w, 21, 3])
    lmap = tf.reshape(dmap, [-1, h, w, 21, 3])
  return hmap, dmap, lmap, masks


def uvz_net(inputs, scope, training):
  with tf.variable_scope(scope):
    feature_layers = resnet50(inputs, scope, training)

    hmap = net_2d(feature_layers[-1], 'hmap', training, 21)
    feature_pack = tf.concat([feature_layers[-1], hmap], -1)

    zmap = net_2d(feature_pack, 'zmap', training, 21)
    feature_pack = tf.concat([feature_pack, zmap], -1)

    masks = net_seg_aux(feature_pack, feature_layers[:-1], 'mask', training)

  return hmap, zmap, masks
