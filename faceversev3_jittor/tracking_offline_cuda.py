import cv2
import os
import numpy as np
import time
import jittor as jt
import argparse
import threading
from queue import Queue, Empty
import logging
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

from util_functions import get_length, ply_from_array_color
from data_reader import OfflineReader
import faceverse_cuda.losses as losses
from faceverse_cuda import get_faceverse

os.environ['JT_USE_CUDA_CACHE'] = '1'
os.environ['JT_CACHE_DIR'] = '../../root/.cache/jittor'
jt.flags.use_cuda = 1

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchOfflineReader:
    def __init__(self, video_path, tar_size, image_size, crop_size, batch_size, skip_frames=0):
        self.reader = OfflineReader(
            video_path, tar_size, image_size, crop_size, skip_frames)
        self.batch_size = batch_size

    def get_batch(self):
        batch = []
        for _ in range(self.batch_size):
            data = self.reader.get_data()
            if not data[0]:  # If not detected
                if not data[1]:  # If no align data
                    continue
                else:
                    break
            batch.append(data)
        return batch


class FaceVerseModel(jt.nn.Module):
    def __init__(self, batch_size, focal, img_size):
        super(FaceVerseModel, self).__init__()
        # ... (other initialization code remains the same)
        self.id_dims = 150
        self.exp_dims = 52
        self.tex_dims = 251

    def split_coeffs(self, coeffs):
        logger.debug(f"Coeffs shape in split_coeffs: {coeffs.shape}")

        # Handle 2D input
        if len(coeffs.shape) == 2:
            coeffs = coeffs[0]  # Take the first row if it's 2D

        # Ensure coeffs is 1D
        if len(coeffs.shape) != 1:
            raise ValueError(f"Expected 1D tensor, got shape {coeffs.shape}")

        try:
            id_coeff = coeffs[:self.id_dims]
            exp_coeff = coeffs[self.id_dims:self.id_dims + self.exp_dims]
            tex_coeff = coeffs[self.id_dims +
                               self.exp_dims:self.id_dims + self.exp_dims + self.tex_dims]
            rot_coeff = coeffs[self.id_dims + self.exp_dims +
                               self.tex_dims:self.id_dims + self.exp_dims + self.tex_dims + 3]
            gamma_coeff = coeffs[self.id_dims + self.exp_dims + self.tex_dims +
                                 3:self.id_dims + self.exp_dims + self.tex_dims + 30]
            trans_coeff = coeffs[self.id_dims + self.exp_dims + self.tex_dims +
                                 30:self.id_dims + self.exp_dims + self.tex_dims + 33]
            eye_coeff = coeffs[self.id_dims + self.exp_dims + self.tex_dims +
                               33:self.id_dims + self.exp_dims + self.tex_dims + 35]
        except Exception as e:
            logger.error(f"Error in split_coeffs: {str(e)}")
            logger.error(f"Coeffs shape: {coeffs.shape}")
            logger.error(f"self.id_dims: {self.id_dims}")
            logger.error(f"self.exp_dims: {self.exp_dims}")
            logger.error(f"self.tex_dims: {self.tex_dims}")
            raise

        return id_coeff, exp_coeff, tex_coeff, rot_coeff, gamma_coeff, trans_coeff, eye_coeff

    # ... (rest of the FaceVerseModel class remains the same)


class Tracking(threading.Thread):
    def __init__(self, args):
        super(Tracking, self).__init__()
        self.args = args
        self.fvm, self.fvd = get_faceverse(batch_size=args.batch_size, focal=int(
            1315 / 512 * args.tar_size), img_size=args.tar_size)
        self.lm_weights = losses.get_lm_weights()
        self.offreader = BatchOfflineReader(
            args.input, args.tar_size, args.image_size, args.crop_size, args.batch_size, args.skip_frames)
        self.thread_lock = threading.Lock()
        self.frame_ind = 0
        self.thread_exit = False
        self.scale = 0
        self.queue = Queue(maxsize=100)

    @staticmethod
    @jt.no_grad()
    def eyes_refine(eye_coeffs):
        return jt.where(eye_coeffs > 0.4, (eye_coeffs - 0.4) * 2 + 0.4, eye_coeffs)

    def run(self):
        optimizer = jt.optim.Adam(
            self.fvm.parameters(), lr=5e-3, betas=(0.5, 0.9))
        start_time = time.time()

        try:
            while not self.thread_exit:
                batch_data = self.offreader.get_batch()
                if not batch_data:
                    self.thread_exit = True
                    break

                batch_align = np.stack([data[1] for data in batch_data])
                batch_lms = np.stack([data[2] for data in batch_data])

                lms = jt.array(batch_lms, dtype=jt.float32).stop_grad()
                img_tensor = jt.array(
                    batch_align, dtype=jt.float32).stop_grad().transpose((0, 3, 1, 2))

                num_iters = 200 if self.frame_ind == 0 else 10

                scale = ((batch_lms - batch_lms.mean(1, keepdims=True))
                         ** 2).mean() ** 0.5
                if self.scale != 0:
                    self.fvm.trans_tensor[:, 2] = (
                        self.fvm.trans_tensor[:, 2] + self.fvm.camera_pos[0, 0, 2]) * self.scale / scale - self.fvm.camera_pos[0, 0, 2]
                self.scale = scale

                for _ in range(num_iters):
                    pred_dict = self.fvm(
                        self.fvm.get_packed_tensors(), render=False)
                    lm_loss_val = losses.lm_loss(
                        pred_dict['lms_proj'], lms, self.lm_weights, img_size=self.args.tar_size)
                    exp_reg_loss = losses.get_l2(
                        self.fvm.exp_tensor[:, 40:]) + losses.get_l2(self.fvm.exp_tensor[:, :40])
                    loss = lm_loss_val * self.args.lm_loss_w + exp_reg_loss * self.args.exp_reg_w

                    optimizer.zero_grad()
                    optimizer.backward(loss)
                    optimizer.step()
                    self.fvm.exp_tensor[self.fvm.exp_tensor < 0] *= 0

                with jt.no_grad():
                    coeffs = self.fvm.get_packed_tensors()
                    logger.debug(
                        f"Coeffs shape before eyes_refine: {coeffs.shape}")
                    coeffs[:, self.fvm.id_dims + 8:self.fvm.id_dims + 10] = self.eyes_refine(
                        coeffs[:, self.fvm.id_dims + 8:self.fvm.id_dims + 10])
                    logger.debug(
                        f"Coeffs shape after eyes_refine: {coeffs.shape}")

                    if self.frame_ind == 0:
                        try:
                            id_c, exp_c, tex_c, rot_c, gamma_c, trans_c, eye_c = self.fvm.split_coeffs(
                                coeffs)
                            np.savetxt(os.path.join(
                                self.args.res_folder, 'id.txt'), id_c.numpy(), fmt='%.3f')
                            np.savetxt(os.path.join(
                                self.args.res_folder, 'exp.txt'), exp_c.numpy(), fmt='%.3f')
                        except Exception as e:
                            logger.error(f"Error in split_coeffs: {str(e)}")
                            logger.error(f"Coeffs shape: {coeffs.shape}")
                            raise

                    pred_dict = self.fvm(coeffs, render=True, surface=True,
                                         use_color=True, render_uv=self.args.save_for_styleavatar)
                    rendered_img = jt.clamp(pred_dict['rendered_img'].transpose(
                        (0, 2, 3, 1)), 0, 255).uint8().numpy()

                    if self.args.save_for_styleavatar:
                        uv_img = jt.clamp(pred_dict['uv_img'].transpose(
                            (0, 2, 3, 1)), 0, 255).uint8().numpy()

                    for i, (frame_num, outimg, _, _, _) in enumerate(batch_data):
                        if self.args.save_for_styleavatar:
                            drive_img = np.concatenate(
                                [batch_align[i], rendered_img[i, :, :, :3], uv_img[i, :, :, :3]], axis=1)
                        else:
                            drive_img = np.concatenate(
                                [batch_align[i], rendered_img[i, :, :, :3]], axis=1)
                        self.queue.put((frame_num, outimg, drive_img))

                self.frame_ind += len(batch_data)
                if self.frame_ind % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = self.frame_ind / elapsed
                    logger.info(
                        f"Processed {self.frame_ind} frames. FPS: {fps:.2f}")

        except Exception as e:
            logger.error(f"Error in Tracking thread: {str(e)}")
            self.thread_exit = True

        self.thread_exit = True


def process_output(args, frame_num, outimg, drive_img, out_video, sess):
    try:
        out_video.write(cv2.cvtColor(drive_img, cv2.COLOR_RGB2BGR))

        if args.save_for_styleavatar:
            cv2.imwrite(os.path.join(args.res_folder, 'image',
                        f'{frame_num:06d}.png'), cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.res_folder, 'render', f'{frame_num:06d}.png'), cv2.cvtColor(
                drive_img[:, args.tar_size:args.tar_size*2], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.res_folder, 'uv', f'{frame_num:06d}.png'), cv2.cvtColor(
                drive_img[:, args.tar_size*2:], cv2.COLOR_RGB2BGR))

            if args.crop_size != 1024:
                mask_in = cv2.resize(cv2.cvtColor(
                    outimg, cv2.COLOR_RGB2BGR), (1024, 1024))
            else:
                mask_in = cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR)

            pha = sess.run(
                ['out'], {'src': mask_in[None, :, :, :].astype(np.float32)})

            if args.crop_size != 1024:
                mask_out = cv2.resize(pha[0][0, 0].astype(
                    np.uint8), (args.crop_size, args.crop_size))
            else:
                mask_out = pha[0][0, 0].astype(np.uint8)

            cv2.imwrite(os.path.join(args.res_folder, 'back',
                        f'{frame_num:06d}.png'), mask_out)
    except Exception as e:
        logger.error(f"Error in process_output: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FaceVerse offline tracker (high throughput)")
    parser.add_argument('--input', type=str, required=True,
                        help='input video path')
    parser.add_argument('--res_folder', type=str,
                        required=True, help='output directory')
    parser.add_argument('--id_folder', type=str,
                        default=None, help='id directory')
    parser.add_argument('--first_frame_is_neutral', action='store_true',
                        help='only if the first frame is neutral expression')
    parser.add_argument('--smooth', action='store_true',
                        help='smooth between 3 frames')
    parser.add_argument('--use_dr', action='store_true',
                        help='Can only be used on linux system.')
    parser.add_argument('--save_for_styleavatar', action='store_true',
                        help='Save images and parameters for styleavatar.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of frames to process in each batch')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Skip the first several frames.')
    parser.add_argument('--crop_size', type=int,
                        default=1024, help='size for output image.')
    parser.add_argument('--image_size', type=int,
                        default=1024, help='size for output image.')
    parser.add_argument('--tar_size', type=int, default=256,
                        help='size for rendering window. We use a square window.')
    parser.add_argument('--lm_loss_w', type=float,
                        default=1e3, help='weight for landmark loss')
    parser.add_argument('--rgb_loss_w', type=float,
                        default=1e-2, help='weight for rgb loss')
    parser.add_argument('--id_reg_w', type=float, default=3e-2,
                        help='weight for id coefficient regularizer')
    parser.add_argument('--rt_reg_w', type=float,
                        default=3e-2, help='weight for rt regularizer')
    parser.add_argument('--exp_reg_w', type=float, default=3e-3,
                        help='weight for expression coefficient regularizer')
    parser.add_argument('--tex_reg_w', type=float, default=3e-3,
                        help='weight for texture coefficient regularizer')
    parser.add_argument('--tex_w', type=float, default=1,
                        help='weight for texture reflectance loss.')

    args = parser.parse_args()

    os.makedirs(args.res_folder, exist_ok=True)
    if args.save_for_styleavatar:
        os.makedirs(os.path.join(args.res_folder, 'image'), exist_ok=True)
        os.makedirs(os.path.join(args.res_folder, 'uv'), exist_ok=True)
        os.makedirs(os.path.join(args.res_folder, 'render'), exist_ok=True)
        os.makedirs(os.path.join(args.res_folder, 'back'), exist_ok=True)

    tracking = Tracking(args)
    tracking.start()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    if args.save_for_styleavatar:
        out_video = cv2.VideoWriter(os.path.join(args.res_folder, 'track.mp4'),
                                    fourcc, tracking.offreader.reader.fps, (args.tar_size * 3, args.tar_size))
    else:
        out_video = cv2.VideoWriter(os.path.join(args.res_folder, 'track.mp4'),
                                    fourcc, tracking.offreader.reader.fps, (args.tar_size * 2, args.tar_size))

    sess = ort.InferenceSession('data/rvm_1024_1024_32.onnx',
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    start_time = time.time()
    frame_count = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        while not tracking.thread_exit or not tracking.queue.empty():
            try:
                frame_num, outimg, drive_img = tracking.queue.get(timeout=1)
                executor.submit(process_output, args, frame_num,
                                outimg, drive_img, out_video, sess)
                frame_count += 1
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                break

    tracking.join()
    out_video.release()

    total_time = time.time() - start_time
    fps = frame_count / total_time
    logger.info(
        f'Processing completed. Total frames: {frame_count}, Time taken: {total_time:.2f} seconds, FPS: {fps:.2f}')
