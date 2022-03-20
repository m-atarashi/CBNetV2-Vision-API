from os import makedirs
from os.path import basename, splitext

import ffmpeg
import time
import numpy as np
from PIL import Image
from numba import jit, void

from CBNetV2 import mydemo, mydemo_batch, mydemo_numba


video_path = '../data/inputs/shelving_short_02.mp4'
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
    frames = [frame for frame in frames][30:60]
    return frames


def main():
    all_frames = extract_frames(video_path)
    model = mydemo_batch.load_model()

    step = inference_batch_size
    for slice_start in range(0, len(all_frames), step):
        frames = all_frames[slice_start : slice_start + step]
        # coords: list(n, 5), type: float. masks: list(n, height, width), type: bool 
        results = mydemo_batch.inference(frames, model)

        for i in range(step):
            output_dir = f'{output_root}/{splitext(basename(video_path))[0]}/frame_{str(i + slice_start).zfill(8)}/'
            makedirs(output_dir, exist_ok = True)

            mydemo_batch.save_masked_image(frames[i], results[i], score_thr=0.3, output_dir=output_dir)


def main2():
    model  = mydemo_numba.load_model()
    frames = extract_frames(video_path)
    for i, frame in enumerate(frames):
        output_dir = f'{output_root}/{splitext(basename(video_path))[0]}/frame_{str(i).zfill(8)}/'
        makedirs(output_dir, exist_ok = True)

        coords, masks = mydemo_numba.inference(frame, model)
        coords = np.array(coords)
        masks  = np.array(masks)
        mydemo_numba.save_masked_image(frame, coords, masks, score_thr=0.3, output_dir=output_dir)


if __name__ == '__main__':
    t_s = time.time()
    main2()
    t_e = time.time()
    print(f'Processing time: {float(t_e-t_s)}[s]')