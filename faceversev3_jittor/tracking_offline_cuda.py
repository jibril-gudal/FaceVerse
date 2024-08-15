from util_functions import get_length, ply_from_array_color
from data_reader import OfflineReader
import faceverse_cuda.losses as losses
from faceverse_cuda import get_faceverse
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


image_queue = Queue(maxsize=30)  # Limit queue size to prevent memory issues


class Tracking(threading.Thread):
    def __init__(self, args):
        super(Tracking, self).__init__()
        self.args = args
        self.fvm, self.fvd = get_faceverse(batch_size=1, focal=int(
            1315 / 512 * self.args.tar_size), img_size=self.args.tar_size)
        self.lm_weights = losses.get_lm_weights()
        self.offreader = OfflineReader(
            args.input, args.tar_size, args.image_size, args.crop_size, skip_frames=args.skip_frames)
        self.thread_lock = threading.Lock()
        self.frame_ind = 0
        self.thread_exit = False
        self.scale = 0

    def eyes_refine(self, eye_coeffs):
        for j in range(2):
            if eye_coeffs[0, j] > 0.4:
                eye_coeffs[0, j] = (eye_coeffs[0, j] - 0.4) * 2 + 0.4
        return eye_coeffs

    def run(self):
        while not self.thread_exit:
            detected, align, lms_detect, outimg, frame_num = self.offreader.get_data()
            if not detected:
                if not align:
                    continue
                else:
                    break

            lms = jt.array(lms_detect[None, :, :],
                           dtype=jt.float32).stop_grad()
            img_tensor = jt.array(
                align[None, :, :, :], dtype=jt.float32).stop_grad().transpose((0, 3, 1, 2))

            if self.frame_ind == 0:
                num_iters_rf, num_iters_nrf = 300, 100
                optimizer = jt.optim.Adam(
                    self.fvm.parameters(), lr=1e-2, betas=(0.8, 0.95))
            else:
                num_iters_rf, num_iters_nrf = 15, 10  # Slightly increased for better accuracy
                optimizer = jt.optim.Adam(
                    self.fvm.parameters(), lr=5e-3, betas=(0.5, 0.9))

            scale = ((lms_detect - lms_detect.mean(0)) ** 2).mean() ** 0.5
            if self.scale != 0:
                self.fvm.trans_tensor[0, 2] = (
                    self.fvm.trans_tensor[0, 2] + self.fvm.camera_pos[0, 0, 2]) * self.scale / scale - self.fvm.camera_pos[0, 0, 2]
            self.scale = scale

            # Combined fitting loop
            for i in range(num_iters_rf + num_iters_nrf):
                pred_dict = self.fvm(
                    self.fvm.get_packed_tensors(), render=(i >= num_iters_rf))
                lm_loss_val = losses.lm_loss(
                    pred_dict['lms_proj'], lms, self.lm_weights, img_size=self.args.tar_size)
                exp_reg_loss = losses.get_l2(
                    self.fvm.exp_tensor[:, 40:]) + losses.get_l2(self.fvm.exp_tensor[:, :40])
                loss = lm_loss_val * self.args.lm_loss_w + exp_reg_loss * self.args.exp_reg_w

                if i >= num_iters_rf:
                    photo_loss_val = losses.photo_loss(
                        pred_dict['rendered_img'][:, :3], img_tensor)
                    loss += photo_loss_val * self.args.rgb_loss_w

                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                self.fvm.exp_tensor[self.fvm.exp_tensor < 0] *= 0

            # Generate output
            with jt.no_grad():
                coeffs = self.fvm.get_packed_tensors().detach()
                coeffs[:, self.fvm.id_dims + 8:self.fvm.id_dims + 10] = self.eyes_refine(
                    coeffs[:, self.fvm.id_dims + 8:self.fvm.id_dims + 10])

                if self.frame_ind == 0:
                    id_c, exp_c, tex_c, rot_c, gamma_c, trans_c, eye_c = self.fvm.split_coeffs(
                        coeffs)
                    np.savetxt(os.path.join(self.args.output,
                               'id.txt'), id_c[0].numpy(), fmt='%.3f')
                    np.savetxt(os.path.join(self.args.output,
                               'exp.txt'), exp_c[0].numpy(), fmt='%.3f')

                pred_dict = self.fvm(
                    coeffs, render=True, surface=True, use_color=True, render_uv=True)
                rendered_img = np.clip(pred_dict['rendered_img'].transpose(
                    (0, 2, 3, 1)).numpy(), 0, 255)[0, :, :, :3].astype(np.uint8)
                uv_img = np.clip(pred_dict['uv_img'].transpose((0, 2, 3, 1)).numpy(), 0, 255)[
                    0, :, :, :3].astype(np.uint8)
                drive_img = np.concatenate(
                    [align, rendered_img, uv_img], axis=1)

                image_queue.put(
                    (frame_num, outimg, drive_img, rendered_img, uv_img))

            self.frame_ind += 1
            if self.frame_ind % 10 == 0:
                print(f'Processed: {self.frame_ind} frames')

        self.thread_exit = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FaceVerse offline tracker (optimized for StyleAvatar)")
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

    # Load the background removal model
    sess = ort.InferenceSession('data/rvm_1024_1024_32.onnx')

    start_time = time.time()
    frame_count = 0

    while True:
        if tracking.thread_exit and image_queue.empty():
            break

        if not image_queue.empty():
            frame_num, outimg, drive_img, rendered_img, uv_img = image_queue.get()
            out_video.write(cv2.cvtColor(drive_img, cv2.COLOR_RGB2BGR))

            # Save individual frames and components
            cv2.imwrite(os.path.join(args.output, 'image', f'{frame_num:06d}.png'), cv2.cvtColor(
                outimg, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.output, 'render', f'{frame_num:06d}.png'), cv2.cvtColor(
                rendered_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.output, 'uv', f'{frame_num:06d}.png'), cv2.cvtColor(
                uv_img, cv2.COLOR_RGB2BGR))

            # Generate and save background mask
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

            cv2.imwrite(os.path.join(args.output, 'back',
                        f'{frame_num:06d}.png'), mask_out)

            frame_count += 1

        time.sleep(0.001)  # Prevent busy-waiting

    tracking.join()
    out_video.release()

    total_time = time.time() - start_time
    fps = frame_count / total_time
    print(
        f'Processing completed. Total frames: {frame_count}, Time: {total_time:.2f} seconds, FPS: {fps:.2f}')
