from data_reader import OfflineReader
import faceverse_cuda.losses as losses
from faceverse_cuda import get_faceverse
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from queue import Queue
import threading
import argparse
import cv2
import os
import numpy as np
import time
import jittor as jt

jt.flags.use_cuda = 1
jt.set_global_seed(0)

# Use a thread-safe queue with a larger buffer
image_queue = Queue(maxsize=100)


class Tracking(threading.Thread):
    def __init__(self, args):
        super(Tracking, self).__init__()
        self.args = args
        self.fvm, self.fvd = get_faceverse(batch_size=args.batch_size, focal=int(
            1315 / 512 * args.tar_size), img_size=args.tar_size)
        self.lm_weights = losses.get_lm_weights()
        self.offreader = OfflineReader(
            args.input, args.tar_size, args.image_size, args.crop_size, skip_frames=args.skip_frames)
        self.thread_lock = threading.Lock()
        self.frame_ind = 0
        self.thread_exit = False
        self.scale = 0
        self.prev_coeffs = None

    @staticmethod
    @jt.no_grad()
    def eyes_refine(eye_coeffs):
        return jt.where(eye_coeffs > 0.4, (eye_coeffs - 0.4) * 2 + 0.4, eye_coeffs)

    def run(self):
        optimizer = jt.optim.Adam(
            self.fvm.parameters(), lr=5e-3, betas=(0.5, 0.9))

        while not self.thread_exit:
            batch_data = [self.offreader.get_data()
                          for _ in range(self.args.batch_size)]
            if not batch_data[0][0]:  # If first item is not detected, end of video
                break

            batch_align = np.stack([data[1] for data in batch_data])
            batch_lms = np.stack([data[2] for data in batch_data])

            lms = jt.array(batch_lms, dtype=jt.float32).stop_grad()
            img_tensor = jt.array(
                batch_align, dtype=jt.float32).stop_grad().transpose((0, 3, 1, 2))

            if self.frame_ind == 0:
                num_iters = 200
            else:
                num_iters = 5  # Extremely aggressive iteration reduction

            scale = ((batch_lms - batch_lms.mean(1, keepdims=True))
                     ** 2).mean() ** 0.5
            if self.scale != 0:
                self.fvm.trans_tensor[:, 2] = (
                    self.fvm.trans_tensor[:, 2] + self.fvm.camera_pos[0, 0, 2]) * self.scale / scale - self.fvm.camera_pos[0, 0, 2]
            self.scale = scale

            if self.prev_coeffs is not None:
                self.fvm.set_packed_tensors(self.prev_coeffs)

            # Simplified fitting loop
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

            # Generate output
            with jt.no_grad():
                coeffs = self.fvm.get_packed_tensors().detach()
                print(f"Coeffs shape: {coeffs.shape}")  # Debug print

                coeffs[:, self.fvm.id_dims + 8:self.fvm.id_dims + 10] = self.eyes_refine(
                    coeffs[:, self.fvm.id_dims + 8:self.fvm.id_dims + 10])
                self.prev_coeffs = coeffs.clone()

                if self.frame_ind == 0:
                    id_c, exp_c, tex_c, rot_c, gamma_c, trans_c, eye_c = self.fvm.split_coeffs(
                        coeffs[0])
                    np.savetxt(os.path.join(self.args.output,
                               'id.txt'), id_c.numpy(), fmt='%.3f')
                    np.savetxt(os.path.join(self.args.output,
                               'exp.txt'), exp_c.numpy(), fmt='%.3f')

                pred_dict = self.fvm(
                    coeffs, render=True, surface=True, use_color=True, render_uv=True)
                rendered_img = np.clip(pred_dict['rendered_img'].transpose(
                    (0, 2, 3, 1)).numpy(), 0, 255).astype(np.uint8)
                uv_img = np.clip(pred_dict['uv_img'].transpose(
                    (0, 2, 3, 1)).numpy(), 0, 255).astype(np.uint8)

                for i in range(self.args.batch_size):
                    frame_num, outimg, _, _, _ = batch_data[i]
                    drive_img = np.concatenate(
                        [batch_align[i], rendered_img[i, :, :, :3], uv_img[i, :, :, :3]], axis=1)
                    image_queue.put(
                        (frame_num, outimg, drive_img, rendered_img[i, :, :, :3], uv_img[i, :, :, :3]))

            self.frame_ind += self.args.batch_size
            if self.frame_ind % 50 == 0:
                print(f'Processed: {self.frame_ind} frames')

        self.thread_exit = True


def process_output(args, frame_num, outimg, drive_img, rendered_img, uv_img, out_video, sess):
    out_video.write(cv2.cvtColor(drive_img, cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(args.output, 'image',
                f'{frame_num:06d}.png'), cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output, 'render', f'{frame_num:06d}.png'), cv2.cvtColor(
        rendered_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output, 'uv',
                f'{frame_num:06d}.png'), cv2.cvtColor(uv_img, cv2.COLOR_RGB2BGR))

    if args.crop_size != 1024:
        mask_in = cv2.resize(cv2.cvtColor(
            outimg, cv2.COLOR_RGB2BGR), (1024, 1024))
    else:
        mask_in = cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR)

    pha = sess.run(['out'], {'src': mask_in[None, :, :, :].astype(np.float32)})

    if args.crop_size != 1024:
        mask_out = cv2.resize(pha[0][0, 0].astype(
            np.uint8), (args.crop_size, args.crop_size))
    else:
        mask_out = pha[0][0, 0].astype(np.uint8)

    cv2.imwrite(os.path.join(args.output, 'back',
                f'{frame_num:06d}.png'), mask_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FaceVerse offline tracker (fully optimized)")
    parser.add_argument('--input', type=str, required=True,
                        help='input video path')
    parser.add_argument('--output', type=str, required=True,
                        help='output directory')
    parser.add_argument('--tar_size', type=int, default=256,
                        help='size for rendering window')
    parser.add_argument('--image_size', type=int,
                        default=1024, help='size for output image')
    parser.add_argument('--crop_size', type=int,
                        default=1024, help='size for output image')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Skip the first several frames')
    parser.add_argument('--lm_loss_w', type=float,
                        default=1e3, help='weight for landmark loss')
    parser.add_argument('--rgb_loss_w', type=float,
                        default=1e-2, help='weight for rgb loss')
    parser.add_argument('--exp_reg_w', type=float, default=1e-3,
                        help='weight for expression coefficient regularizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of frames to process in each batch')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'uv'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'back'), exist_ok=True)

    tracking = Tracking(args)
    tracking.start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(os.path.join(args.output, 'tracked.mp4'),
                                fourcc, tracking.offreader.fps, (args.tar_size * 3, args.tar_size))

    sess = ort.InferenceSession('data/rvm_1024_1024_32.onnx')

    start_time = time.time()
    frame_count = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        while not (tracking.thread_exit and image_queue.empty()):
            if not image_queue.empty():
                frame_data = image_queue.get()
                executor.submit(process_output, args, *
                                frame_data, out_video, sess)
                frame_count += 1

    tracking.join()
    out_video.release()

    total_time = time.time() - start_time
    fps = frame_count / total_time
    print(
        f'Processing completed. Total frames: {frame_count}, Time: {total_time:.2f} seconds, FPS: {fps:.2f}')
