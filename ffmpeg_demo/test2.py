import ffmpeg
import numpy as np
from os import makedirs
from os.path import basename
from PIL import Image
from CBNetV2 import mydemo, mydemo_batch


video_path = '../data/inputs/shelving_short.mp4'
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
    frames = [frame for frame in frames]
    return frames


def main():
    all_frames = extract_frames(video_path)
    model = mydemo_batch.load_model()

    step_size = 4
    for slice_start in range(0, len(all_frames), step_size):
        frames = all_frames[slice_start : slice_start + step_size]
        results = mydemo_batch.inference(frames, model)
        for i in range(len(frames)):
            output_dir = f'{output_root}/{basename(video_path)}/frame_{str(i + slice_start).zfill(8)}/'
            makedirs(output_dir)
            mydemo_batch.save_masked_image(frames[i], results[i], score_thr=0.3, output_dir=output_dir)


if __name__ == '__main__':
    main()