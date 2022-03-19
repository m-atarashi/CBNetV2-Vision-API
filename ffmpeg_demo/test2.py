import ffmpeg
import numpy as np
from PIL import Image
from CBNetV2 import mydemo

input_path = '../demo_data/inputs/shelving_short.mp4'

height = ffmpeg.probe(input_path)['streams'][0]['height']
width = ffmpeg.probe(input_path)['streams'][0]['width']

output = (
    ffmpeg
    .input(input_path)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run(capture_stdout=True)
)

frames = np.frombuffer(output, np.uint8).reshape([-1, height, width, 3])

