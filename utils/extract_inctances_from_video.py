from os import makedirs
from os.path import basename, splitext

import ffmpeg
import time
import numpy as np
from PIL import Image

from CBNetV2 import mydemo


video_path = '../data/inputs/shelving_short_02.mp4'
output_root = '../data/outputs'


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
    frames = extract_frames(video_path)
    model = mydemo.load_model()

    for i, frame in enumerate(frames):
        print(f'frame: {i+1}')
        result = mydemo.inference(frame, model)

        output_dir = f'{output_root}/{splitext(basename(video_path))[0]}/frame_{str(i + slice_start).zfill(8)}/'
        makedirs(output_dir, exist_ok = True)
        mydemo.save_instances(frame, result, score_thr=0.3, output_dir=output_dir)


if __name__ == '__main__':
    t_s = time.time()
    main()
    t_e = time.time()
    print(f'Processing time: {float(t_e-t_s)}[s]')