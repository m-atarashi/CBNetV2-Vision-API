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

        output_dir = f'{output_root}/{splitext(basename(video_path))[0]}/frame_{str(i).zfill(8)}/'
        makedirs(output_dir, exist_ok = True)
        mydemo.save_instances(frame, result, score_thr=0.3, output_dir=output_dir)


if __name__ == '__main__':
    t_s = time.time()
    main()
    t_e = time.time()
    print(f'Processing time: {float(t_e-t_s)}[s]')






# 徒労
# def main_alter():
#     frames = extract_frames(video_path)
#     model = mydemo_batch_alter.load_model()

#     step = inference_batch_size
#     for slice_start in range(0, len(frames), step):
#         print(f'frames: {slice_start}~{slice_start+step-1}')
#         batch_frames = frames[slice_start : slice_start + step]

#         batch_bboxes, batch_masks = mydemo_batch_alter.inference(batch_frames, model)

#         for i in range(step):
#             if (1 + i + slice_start > len(frames)):
#                 return
            
#             output_dir = f'{output_root}/{splitext(basename(video_path))[0]}/frame_{str(1 + i + slice_start).zfill(8)}/'
#             makedirs(output_dir, exist_ok=True)

#             masks = np.array(batch_masks[i])
#             masks = masks.reshape(*masks.shape, 1).astype(np.uint8)
#             coords = batch_bboxes[i][:,:4].astype(np.uint16)
#             scores = batch_bboxes[i][:,4]

#             mydemo_batch_alter.save_instances(batch_frames[i], coords, scores, masks, score_thr=0.3, output_dir=output_dir)

