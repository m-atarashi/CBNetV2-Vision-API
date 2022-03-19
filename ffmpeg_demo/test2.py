import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = '../demo_data/inputs/shelving_short.mp4'

height = ffmpeg.probe(path)['streams'][0]['height']
width = ffmpeg.probe(path)['streams'][0]['width']


out, _ = (
    ffmpeg
    .input(path)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run(capture_stdout=True)
)

arr = (
    np
    .frombuffer(out, np.uint8)
    .reshape([-1, height, width, 3])
)


print(arr.shape)
Image.fromarray(arr[1]).save('../demo_data/outputs/test2_1.jpg')
Image.fromarray(arr[300]).save('../demo_data/outputs/test2_2.jpg')
Image.fromarray(arr[600]).save('../demo_data/outputs/test2_3.jpg')