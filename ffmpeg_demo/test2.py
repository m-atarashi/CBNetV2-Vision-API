import ffmpeg
import numpy as np
from os.path import mkdir, basename
from PIL import Image
from CBNetV2 import mydemo, mydemo_batch


video_path = '../data/inputs/shelving_short.mp4'
output_root = '../data/outpus'


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
    return frames


def main():
    frames = extract_frames(video_path)
    results = mydemo_batch.inference(frames)
    for i in len(frames):
        output_dir = f'{output_root}/{basename(video_path)}/frame_{str(i).zfill(8)}/'
        mkdir(output_dir)
        mydemo_batch.save_masked_image(frames[i], results[i], score_thr=0.3, output_dir=output_dir)


if __name__ == '__main__':
    main()