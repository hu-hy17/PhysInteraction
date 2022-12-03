import argparse
import base64
import cgi
import json
import socketserver
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# import GPUtil
from GPUtil import GPUtil
from PIL import Image

from network import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='Gpu index to use. [Default: 1]')
parser.add_argument('--port', type=int, default=8080, help='Server port. [Default: 8080]')
FLAGS = parser.parse_args()

tf.logging.set_verbosity('INFO')

try:
    req_gpu = int(FLAGS.gpu)
    existing_gpus = GPUtil.getGPUs()
    existing_gpus = [g.id for g in existing_gpus]
    if req_gpu not in existing_gpus:
        raise ValueError
    os.environ['CUDA_VISIBLE_DEVICES'] = str(req_gpu)
except ValueError:
    tf.logging.error('Check your gpus parameter.')
    exit(1)


def tf_hmap_to_uv(hmap):
    hmap_flat = tf.reshape(hmap, (tf.shape(hmap)[0], -1, tf.shape(hmap)[3]))
    argmax = tf.argmax(hmap_flat, axis=1, output_type=tf.int32)
    argmax_x = argmax // tf.shape(hmap)[2]
    argmax_y = argmax % tf.shape(hmap)[2]
    uv = tf.stack((argmax_x, argmax_y), axis=1)
    uv = tf.transpose(uv, [0, 2, 1])
    return uv


class ModelUVZ_D:
    def __init__(self, model_path):
        self.input_ph = tf.placeholder(tf.float32, [240, 320])
        self.inputs = tf.expand_dims(tf.expand_dims(self.input_ph, 0), -1)
        self.hmap, self.zmap, self.masks = \
            uvz_net(self.inputs, 'deep_depth_hand', False)

        self.uv = tf_hmap_to_uv(self.hmap)
        self.z = tf.gather_nd(
            tf.transpose(self.zmap, [0, 3, 1, 2]), self.uv, batch_dims=2
        )[0]
        self.uv = self.uv[0]
        self.mask = self.masks[-1][0, :, :, 0]
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, model_path)

        self.z_back = []
        self.uv_back = []
        self.mask_back = []
        self.hand_object_back = []
        self.ready = False

    def pipeline_run(self, depth_img, begin_flag):
        if begin_flag > 0:
            z, uv, mask, _, _ = self.process(depth_img)
            self.z_back = z
            self.uv_back = uv
            self.mask_back = mask
            self.hand_object_back = np.uint8((depth_img > 100) * 255)
            self.ready = True
        else:
            # threading.Thread(target=self.thread_run, args=(depth_img,)).start()
            self.thread_run(depth_img)

        # while  self.ready==False:
        #     print("wait for data ready")

        return self.z_back, self.uv_back, self.mask_back, self.hand_object_back

    def thread_run(self, depth_img):
        time1 = time.time()
        z, uv, mask, _, _ = self.process(depth_img)
        time_process = time.time()
        print("process time one frame:%f" % ((time_process - time1) * 1000))

        self.z_back = z
        self.uv_back = uv
        self.mask_back = mask
        self.hand_object_back = np.uint8((depth_img > 100) * 255)
        self.ready = True

    def process(self, depth):
        depth_original = depth / 8  # mm
        depth[depth == 0] = np.nan
        offset = np.nanmin(depth, axis=(0, 1), keepdims=True)
        scale = np.nanmax(depth, axis=(0, 1), keepdims=True) - offset
        depth = (depth - offset) / scale
        depth[np.isnan(depth)] = 0

        z, uv, mask = \
            self.sess.run([self.z, self.uv, self.mask], {self.input_ph: depth})
        z = ((z * scale + offset) / 8)[0]  # mm
        uv *= 8

        z_combined = []
        z_source = []
        for j in range(21):
            if mask[uv[j][0], uv[j][1]] < 0.5:  # object
                z_combined.append(z[j])
                z_source.append('prediction')
            else:
                z_combined.append(depth_original[uv[j][0], uv[j][1]])
                z_source.append('depth map')

        return z, uv, mask, z_combined, z_source


try:

    model_path = "model/JointLearningNN/150.ckpt"
    model = ModelUVZ_D(model_path)

    depth_img = cv2.imread("./org_depth_img_init.png", cv2.IMREAD_UNCHANGED)
    depth_img = np.asarray(depth_img).astype(np.float32)
    hand_object_mask = np.uint8((depth_img > 100) * 255)

    z, uv, mask, z_combined, z_source = model.process(depth_img.copy())

except ModuleNotFoundError:
    tf.logging.error('Model load error')
    exit(1)

frame_id = 0


def build_inference_handler():
    class InferenceHandler(BaseHTTPRequestHandler):

        def __init__(self, *args, **kwargs):
            super(InferenceHandler, self).__init__(*args, **kwargs)
            # self.frame_id=0

        def do_GET(self):
            if not self.ip_check():
                self.send_response(404)
                return
            page = "<form enctype=\"multipart/form-data\" method=\"post\">" \
                   "Input depth map: <input type=\"file\" name=\"depth\"><br>" \
                   "<input type=\"submit\" value=\"submit\">" \
                   "</form>"
            self.send_response(200)
            self.send_header('Content-Type',
                             'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(page.encode('utf8'))

        def do_POST(self):
            global frame_id

            time0 = time.time()
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': self.headers['Content-Type'],
                }
            )

            depth_img = Image.open(form['depth'].file)
            depth_img = np.asarray(depth_img).astype(np.float32)

            begin_flag = int(form['begin'].file.read())
            print("begin_flag:%d\n" % begin_flag)

            time1 = time.time()
            # print("start processing")
            # z, uv, mask, z_combined, z_source = model.process(depth_img.copy())
            z, uv, mask, hand_object_mask = model.pipeline_run(depth_img.copy(), begin_flag)
            if begin_flag == 0:
                model.ready = False

            # print("active thread number:%d\n"%threading.active_count())

            time_process = time.time()
            # print("process time:%dms\n"%((time_process - time1) * 1000))

            frame_id = frame_id + 1

            hand_mask_Pred = np.uint8(mask * 255)
            not_hand_mask_Pred = cv2.bitwise_not(hand_mask_Pred)
            object_mask_Pred = cv2.bitwise_and(not_hand_mask_Pred, hand_object_mask)
            mask_Pred = np.uint8(hand_mask_Pred + object_mask_Pred / 2)

            data_encode = np.array(cv2.imencode('.png', mask_Pred)[1])
            byte_encode = data_encode.tobytes()

            z_expand = np.expand_dims(z, axis=1)
            joints = np.hstack((np.float32(uv), z_expand))
            joints = np.array(np.float32(joints))

            joints_byte = joints.tobytes()

            enc = "UTF-8"
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            buf = {"joints": (base64.b64encode(joints_byte)).decode(),
                   "segment": (base64.b64encode(byte_encode)).decode()}
            encode_char = json.dumps(buf).encode()
            self.wfile.write(encode_char)

            time2 = time.time()
            print('processing time 1: %dms\t time 2: %dms' % ((time_process - time1) * 1000, (time2 - time0) * 1000))

    return InferenceHandler


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == '__main__':
    inference_handler = build_inference_handler()

    server_address = ('0.0.0.0', FLAGS.port)
    server = ThreadedHTTPServer(server_address, inference_handler)
    print('Starting server at %s, use <Ctrl-C> to stop' % (server_address[0] + ':' + str(server_address[1])))
    server.serve_forever()
