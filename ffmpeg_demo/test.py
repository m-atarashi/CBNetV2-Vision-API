import ffmpeg
import numpy as np

input_filename = '../data/inputs/shelving_short.mp4'
output_filename = '../data/outputs/a.mp4'

height = ffmpeg.probe(input_filename)['streams'][0]['height']
width = ffmpeg.probe(input_filename)['streams'][0]['width']

process1 = (
    ffmpeg
    .input(input_filename)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)

process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
    .output(output_filename, pix_fmt='yuv420p')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

while True:
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([height, width, 3])
    )
    out_frame = in_frame * 0.1
    process2.stdin.write(
        out_frame
        .astype(np.uint8)
        .tobytes()
    )

process2.stdin.close()
process1.wait()
process2.wait()