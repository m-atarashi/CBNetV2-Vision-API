import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

input_path = '../demo_data/inputs/shelving_short.mp4'

height = ffmpeg.probe(input_path)['streams'][0]['height']
width = ffmpeg.probe(input_path)['streams'][0]['width']


process = (
    ffmpeg
    .input(input_path)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

frames, process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run(pipe_stdin=True, capture_stdout=True)
)


while True:
    in_bytes = process.stdout.read(width * height * 3)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([-1, height, width, 3])
    )
    process2.stdin.write(in_frame.astype(np.uint8).tobytes())

process2.stdin.close()
process.wait()
process2.wait()

Image.fromarray(frames[1]).save('../demo_data/outputs/test2_1.jpg')
Image.fromarray(frames[300]).save('../demo_data/outputs/test2_2.jpg')
Image.fromarray(frames[600]).save('../demo_data/outputs/test2_3.jpg')