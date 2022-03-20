from os import makedirs
from os.path import basename, splitext

import ffmpeg
import time
import numpy as np
from PIL import Image
from numba import jit

from CBNetV2 import mydemo, mydemo_batch, mydemo_batch_alter


video_path = '../data/inputs/shelving_short_02.mp4'
video_basename = splitext(basename(video_path))[0]
output_root = '../data/outputs'
# batch_size over 2 causes CUDA out of memory lol
inference_batch_size = 2


def extract_frames(video_path):
    height_px = ffmpeg.probe(video_path)['streams'][0]['height']
    width_px  = ffmpeg.probe(video_path)['streams'][0]['width']

    output, _ = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )

    frames = np.frombuffer(output, np.uint8).reshape([-1, height_px, width_px, 3])
    return [frame for frame in frames]


def main():
    all_frames = extract_frames(video_path)
    model = mydemo_batch.load_model()

    step = inference_batch_size
    for slice_start in range(0, len(all_frames), step):
        batch_frames = all_frames[slice_start : slice_start + step]

        results = mydemo_batch.inference(batch_frames, model)
        for i, result in enumerate(results):
            output_dir = f'{output_root}/{splitext(basename(video_path))[0]}/frame_{str(i + slice_start).zfill(8)}/'
            makedirs(output_dir, exist_ok = True)

            mydemo_batch.save_masked_image(batch_frames[i], result, score_thr=0.3, output_dir=output_dir)


def main_alter():
    all_frames = extract_frames(video_path)
    model = mydemo_batch_alter.load_model()

    step = inference_batch_size
    for slice_start in range(0, len(all_frames), step):
        batch_frames = all_frames[slice_start : slice_start + step]
        
        batch_bboxes, batch_masks = mydemo_batch_alter.inference(batch_frames, model)
        batch_masks = np.array(batch_masks, dtype=np.uint8))
        batch_masks = batch_masks.reshape(*batch_masks.shape, 1)
        batch_coords = batch_bboxes.astype(np.uint16)[:][:4]
        batch_scores = batch_bboxes[:][4]

        for i in range(step):
            output_dir = f'{output_root}/{splitext(basename(video_path))[0]}/frame_{str(i + slice_start).zfill(8)}/'
            makedirs(output_dir, exist_ok=True)

            mydemo_batch_alter.save_instances(batch_frames[i], batch_coords[i], batch_scores[i], batch_masks[i], output_dir=output_dir)


if __name__ == '__main__':
    t_s = time.time()
    main_alter()
    t_e = time.time()
    print(f'Processing time: {float(t_e-t_s)}[s]')