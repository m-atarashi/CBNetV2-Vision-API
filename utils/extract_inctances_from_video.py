from os import makedirs
from os.path import basename, splitext

import ffmpeg
import time
import numpy as np
from PIL import Image

from CBNetV2 import mydemo

CBNETV2_HOME = '/home/m-atarashi/CBNetV2'
OUTPUT_ROOT = f'{CBNETV2_HOME}/data/outputs'


def extract_frames(video_path):
    height = ffmpeg.probe(video_path)['streams'][0]['height']
    width  = ffmpeg.probe(video_path)['streams'][0]['width']

    output, _ = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )

    frames = np.frombuffer(output, np.uint8).reshape([-1, height, width, 3])
    return frames


def extract_instances(frames, base_output_dir):
    model = mydemo.load_model()

    for i, frame in enumerate(frames):
        print(f'frame: {i+1}')
        result = mydemo.inference(frame, model)

        output_dir = f'{base_output_dir}/frame_{str(i).zfill(8)}'
        makedirs(output_dir, exist_ok=True)
        mydemo.save_instances(frame, result, score_thr=0.3, output_dir=output_dir)


def main():
    video_path = f'{CBNETV2_HOME}/data/inputs/shelving_short_02.mp4'
    frames = extract_frames(video_path)

    base_output_dir = f'{OUTPUT_ROOT}/{splitext(basename(video_path))[0]}_test'
    extract_instances(frames[:30], base_output_dir)


if __name__ == '__main__':
    t_s = time.time()
    main()
    t_e = time.time()
    print(f'Processing time: {float(t_e-t_s)}[s]')