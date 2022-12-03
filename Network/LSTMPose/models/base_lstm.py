import tensorflow as tf
import tensorflow.contrib as contrib
import math


def pred_net(features,
             confidence,
             seq_lens=None,
             is_training=False,
             is_inference=False,
             is_initial=tf.constant(False, tf.bool, []),
             last_state=None,
             lstm_num_layer=3,
             lstm_num_unit=256):
    """
    Direct regression.
    :param features: angles, tf.float32 [batch_size, num_seq, num_feat], (-pi, pi)
    :param confidence: confidence coefficient, tf.float32 [batch_size, num_seq, num_feat] [0, 1]
    :param is_training:
    :return: output. (-pi, pi)
    """
    assert not (is_inference and is_training)
    assert not (is_inference and last_state is None)
    assert not (not is_inference and seq_lens is None)

    num_feat = features.get_shape()[-1].value

    with tf.variable_scope('pred_net'):
        features = features / math.pi
        if confidence is not None:
            inputs = tf.concat([features, confidence], -1)
        else:
            inputs = features

        if lstm_num_layer == 1:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_num_unit)
        else:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(lstm_num_unit) for _ in range(lstm_num_layer)])
        lstm_cell = contrib.rnn.OutputProjectionWrapper(lstm_cell, num_feat, activation=None)

        if not is_inference:
            # inputs = tf.unstack(inputs, axis=1)
            # outputs, _ = tf.nn.static_rnn(lstm_cell, inputs, dtype=tf.float32)
            # outputs = tf.stack(outputs, 1)
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, seq_lens, dtype=tf.float32, parallel_iterations=128)
            outputs = outputs * math.pi
            outputs = tf.clip_by_value(outputs, -math.pi, math.pi)
            return outputs
        else:
            state = tf.cond(is_initial, lambda: lstm_cell.zero_state(1, tf.float32), lambda: last_state)
            outputs, state = lstm_cell.call(inputs, state)
            outputs = outputs * math.pi
            outputs = tf.clip_by_value(outputs, -math.pi, math.pi)
            return outputs, state


if __name__ == '__main__':
    fake_features = tf.random_normal([1, 25])
    fake_conf = tf.ones([1, 25])
    fake_label = tf.random_normal([1, 25])

    default_state = []
    for _ in range(3):
        default_state.append(tf.nn.rnn_cell.LSTMStateTuple(tf.constant(0, tf.float32, [1, 256]),
                                                           tf.constant(0, tf.float32, [1, 256])))
    fake_outputs = pred_net(fake_features, fake_conf,
                            is_inference=True,
                            last_state=tuple(default_state))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out_val = sess.run(fake_outputs)
        pass
