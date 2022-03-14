from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from os.path import basename, splitext
from glob import glob


config_files = [
    'configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py',
    'configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
]
checkpoint_files = [
    'checkpoints/mask_rcnn_swin_small_patch4_window7.pth',
    'checkpoints/cascade_mask_rcnn_swin_small_patch4_window7.pth'
]

device = 'cuda:0'


def demo(img, model, score_thr=0.3):
    # inference the demo image
    result = inference_detector(model, img)
    output = 'demo/frames/' + splitext(basename(img))[0] + '_output.jpg'
    model.show_result(img, result, score_thr=score_thr, out_file=output)


def main():
    model_index = 1
    # init a detector
    model = init_detector(config_files[model_index], checkpoint_files[model_index], device=device)
    
    imgs = sorted(glob('demo/shelving/*.png'))
    for img in imgs:
        demo(img, model)


if __name__ == "__main__":
    main()