import argparse
import cgi
import importlib
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import GPUtil
import ast

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='Gpu index to use. [Default: 0]')
parser.add_argument('--gpu_allow_growth', type=ast.literal_eval, default=True,
                    help='Gpu memory growth. [Default: True]')
parser.add_argument('--path', type=str, default=None,
                    help='Where to store training summary and ckpt. None for new path. [Default: None]')
parser.add_argument('--model', type=str, default='base_lstm',
                    help='Which model to be used. [Default: base_lstm]')
parser.add_argument('--without_conf', type=ast.literal_eval, default=False,
                    help='If true, train a model that do not take confidence as input. [Default: False]')

parser.add_argument('--dif_training', type=ast.literal_eval, default=False,
                    help='If true, use difference between label and input as training objective. [Default: False]')
parser.add_argument('--num_feat', type=int, default=26,
                    help='Features size. [Default: 26]')
parser.add_argument('--lstm_num_layer', type=int, default=3,
                    help='Layer number of the LSTM. [Default: 3]')
parser.add_argument('--lstm_num_unit', type=int, default=256,
                    help='Unit number of the LSTM. [Default: 256]')

parser.add_argument('--port', type=int, default=8081, help='Server port. [Default: 8081]')
parser.add_argument('--additional_allowed_ip', type=str, default=None, help='One ip. [Default: None]')

FLAGS = parser.parse_args()

FLAGS.model = "lstm_with_encoder"
FLAGS.gpu = '0'  # 1
FLAGS.path = "exp/n22_lstm_encoder_dif0_noise5_conf0_shift0_l3_f256"
FLAGS.without_conf = True
FLAGS.num_feat = 22

tf.logging.set_verbosity('INFO')

try:
    pred_net = importlib.import_module('models.' + FLAGS.model)
    pred_net = pred_net.pred_net
except ModuleNotFoundError:
    tf.logging.error("Model '%s' not found." % FLAGS.model)
    exit(1)

try:
    req_gpus = [int(s) for s in FLAGS.gpu.split()]
    existing_gpus = GPUtil.getGPUs()
    existing_gpus = [g.id for g in existing_gpus]
    for gpu in req_gpus:
        if gpu not in existing_gpus:
            raise ValueError
except ValueError:
    tf.logging.error('Check your gpus parameter.')
    exit(1)


def build_inference_handler(session):
    class InferenceHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.session = session
            super(InferenceHandler, self).__init__(*args, **kwargs)

        def do_GET(self):
            if not self.ip_check():
                self.send_response(404)
                return
            page = "<form enctype=\"multipart/form-data\" method=\"post\">" \
                   "Input pose (with %d features): <input type=\"text\" name=\"pose\"><br>" % FLAGS.num_feat
            if not FLAGS.without_conf:
                page += "Input conf (with %d features): <input type=\"text\" name=\"conf\"><br>" % FLAGS.num_feat
            page += "Pipeline id (to identify input stream): <input type=\"text\" name=\"id\" value=\"1\"><br>" \
                    "<input type=\"submit\" value=\"submit\">" \
                    "</form>"
            self.send_response(200)
            self.send_header('Content-Type',
                             'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(page.encode('utf8'))

        def do_POST(self):

            # if not self.ip_check():
            #     self.send_response(404)
            #     return

            time0 = time.time()
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': self.headers['Content-Type'],
                }
            )

            pose_data = form['pose'].file.read()
            pose_data = pose_data.split()
            pose = [float(p) for p in pose_data]
            assert len(pose) == FLAGS.num_feat
            pose = np.asarray(pose, np.float32)
            if FLAGS.without_conf:
                conf = None
            else:
                conf_data = form['conf'].file.read()
                conf_data = conf_data.split()
                conf = [float(c) for c in conf_data]
                assert len(conf) == FLAGS.num_feat
                conf = np.asarray(conf)
            pipeline_id = int(form['id'].file.read())

            time1 = time.time()

            preds, gpu_id = self.session.run(pose, conf, pipeline_id=pipeline_id)

            self.send_response(200)
            self.send_header('Content-Type',
                             'text/plain')
            self.end_headers()
            preds_str = str(preds[0])
            for i in range(1, len(preds)):
                preds_str += ' ' + str(preds[i])
            self.wfile.write(preds_str.encode('utf8'))

            time2 = time.time()
            print('GPU %d: processing time 1: %dms\t time 2: %dms' % (
                gpu_id, (time1 - time0) * 1000, (time2 - time1) * 1000))

        def ip_check(self, prefix='166.111.8'):
            if self.client_address[0].startswith(prefix):
                return True
            if FLAGS.additional_allowed_ip is not None:
                if self.client_address[0].startswith(FLAGS.additional_allowed_ip):
                    return True
            return False

    return InferenceHandler


class LSTMPoseSession:
    def __init__(self, gpu_ids, keep_sess_time=10):
        assert type(gpu_ids) is list
        assert len(gpu_ids) > 0

        self.workers = [{'gpu_id': gpu_id} for gpu_id in gpu_ids]
        self.worker_num = len(gpu_ids)
        self.worker_pipeline_id = [-1 for _ in range(self.worker_num)]
        self.keep_sess_time = keep_sess_time

        for i in range(self.worker_num):
            self.workers[i]['available'] = True
            self.init_gpu(i)

    def init_gpu(self, worker_id):
        worker = self.workers[worker_id]
        gpu_id = worker['gpu_id']

        tf.logging.info('GPU %d: initializing' % gpu_id)
        with tf.Graph().as_default():
            with tf.device('/device:GPU:%d' % gpu_id):
                if FLAGS.path is None:
                    folders = os.listdir('./exp')
                    if not folders:
                        tf.logging.error('No existing trained dir')
                        exit(1)
                    folders.sort()
                    folder = folders[-1]
                    ckpt_path = os.path.join('./exp', folder, 'ckpt')
                else:
                    if not os.path.exists(FLAGS.path):
                        tf.logging.error('FLAGS.path %s does not exist.' % FLAGS.path)
                        exit(1)
                    ckpt_path = os.path.join(FLAGS.path, 'ckpt')

                worker['input_pose_pl'] = tf.placeholder_with_default(tf.zeros([1, FLAGS.num_feat]),
                                                                      [None, FLAGS.num_feat])
                if not FLAGS.without_conf:
                    worker['input_conf_pl'] = tf.placeholder_with_default(tf.ones([1, FLAGS.num_feat]),
                                                                          [None, FLAGS.num_feat])
                else:
                    worker['input_conf_pl'] = None
                worker['is_initial_pl'] = tf.placeholder_with_default(tf.constant(False, tf.bool, []), [])
                default_state = np.zeros((FLAGS.lstm_num_layer, 2, 1, FLAGS.lstm_num_unit), np.float32)
                worker['state_pl'] = tf.placeholder_with_default(default_state,
                                                                 (FLAGS.lstm_num_layer, 2, 1, FLAGS.lstm_num_unit))

                layers = tf.unstack(worker['state_pl'], axis=0)
                rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(layers[idx][0], layers[idx][1])
                                         for idx in range(FLAGS.lstm_num_layer)])

                worker['preds'], worker['states'] = \
                    pred_net(worker['input_pose_pl'], worker['input_conf_pl'],
                             is_inference=True,
                             is_initial=worker['is_initial_pl'],
                             last_state=rnn_tuple_state,
                             lstm_num_layer=FLAGS.lstm_num_layer,
                             lstm_num_unit=FLAGS.lstm_num_unit)
                if FLAGS.dif_training:
                    # training_labels = labels - features
                    # pred_pose = features + preds
                    worker['preds'] = worker['input_pose_pl'] + worker['preds']

                # Deferential preds
                # worker['preds'] = worker['input_pose_pl'] + worker['preds']

                trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                ref_names = [var.name.replace('pred_net/multi_rnn_cell',
                                              'pred_net/rnn/output_projection_wrapper/multi_rnn_cell')[:-2]
                             for var in trainable_vars]
                ref_names[-2:] = [name.replace('pred_net', 'pred_net/rnn/output_projection_wrapper')
                                  for name in ref_names[-2:]]

                saver = tf.train.Saver(dict(zip(ref_names, trainable_vars)))

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = FLAGS.gpu_allow_growth
            config.allow_soft_placement = True
            sess = tf.Session(config=config)

            if tf.train.get_checkpoint_state(ckpt_path) is not None:
                saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
                tf.logging.info(
                    'GPU %d: successfully restored from %s' % (gpu_id, tf.train.latest_checkpoint(ckpt_path)))
            else:
                tf.logging.error('GPU %d: Cannot find checkpoint file' % gpu_id)
                exit(1)
            worker['session'] = sess

        self.run(None, None, worker_id=worker_id)

    def get_worker_id(self, pipeline_id):
        assert pipeline_id >= 0
        for worker_id in range(self.worker_num):
            if self.worker_pipeline_id[worker_id] == pipeline_id:
                return worker_id
        while True:
            for worker_id in range(self.worker_num):
                if time.time() - self.workers[worker_id]['last_run_time'] > self.keep_sess_time:
                    self.worker_pipeline_id[worker_id] = pipeline_id
                    # clear state
                    self.workers[worker_id]['last_state'] = None
                    return worker_id
                time.sleep(0.001)
            time.sleep(0.01)

    def run(self, pose, conf, worker_id=None, pipeline_id=None):
        assert (worker_id is not None) or (pipeline_id is not None)
        assert (worker_id is None) or (pipeline_id is None)

        # initial run
        if worker_id is not None:
            worker = self.workers[worker_id]
            worker['session'].run(worker['preds'], feed_dict={worker['is_initial_pl']: True})
            # make sure it can be used once initializing run finished
            worker['last_run_time'] = time.time() - self.keep_sess_time
            worker['last_state'] = None
            return

        # real run
        worker_id = self.get_worker_id(pipeline_id)
        worker = self.workers[worker_id]
        worker['last_run_time'] = time.time()

        feed_dict = {worker['input_pose_pl']: np.expand_dims(pose, 0)}
        if not FLAGS.without_conf:
            feed_dict[worker['input_conf_pl']] = np.expand_dims(conf, 0)
        # frame 0
        if worker['last_state'] is None:
            feed_dict[worker['is_initial_pl']] = True
        else:
            feed_dict[worker['is_initial_pl']] = False
            feed_dict[worker['state_pl']] = worker['last_state']
        pred_pose, state_val = worker['session'].run([worker['preds'], worker['states']], feed_dict=feed_dict)
        if FLAGS.lstm_num_layer == 1:
            state_val = np.expand_dims(state_val, 0)
        worker['last_state'] = np.asarray(state_val, np.float32)
        worker['last_run_time'] = time.time()
        pred_pose = pred_pose[0]

        return pred_pose, worker['gpu_id']


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == '__main__':
    lstm_pose_session = LSTMPoseSession(gpu_ids=req_gpus)
    inference_handler = build_inference_handler(lstm_pose_session)

    server_address = ('0.0.0.0', FLAGS.port)
    server = ThreadedHTTPServer(server_address, inference_handler)
    print('Starting server at %s, use <Ctrl-C> to stop' % (server_address[0] + ":" + str(server_address[1])))
    server.serve_forever()
