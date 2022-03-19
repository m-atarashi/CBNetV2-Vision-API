import numpy as np
from PIL import Image

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

CBNetV2_HOME = '/home/m-atarashi/CBNetV2/CBNetV2/'

configs = [
    f'{CBNetV2_HOME}configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py',
    f'{CBNetV2_HOME}configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py'
]
checkpoints = [
    f'{CBNetV2_HOME}checkpoints/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth',
    f'{CBNetV2_HOME}checkpoints/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth'
]

device = 'cuda:0'


def inference(images, config_file, checkpoint_file, score_thr=0.3):
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    results = inference_detector(model, images)
    return results


def save_masked_image(img_path, result, score_thr=0.3):
    img = np.array(Image.open(img_path))

    for class_index, mask_by_class in enumerate(result[1]):
        if not mask_by_class:
            continue
        for instance_index, mask_by_insetance in enumerate(mask_by_class):
            if result[0][class_index][instance_index][4] < score_thr:
                continue

            mask = mask_by_insetance.astype(np.uint8)
            if mask.shape[:2] != img.shape[:2]:
                mask = np.array(Image.fromarray(mask).resize((img.shape[1], img.shape[0]), Image.BICUBIC))
            mask = mask.reshape(*mask.shape, 1)

            dst = img * mask
            scale = mask.shape[0]/mask_by_insetance.astype(np.uint8).shape[0]
            coord = (scale*np.array(result[0][class_index][instance_index])).astype(np.uint16)[:4]
            dst = dst[coord[1]:coord[3], coord[0]:coord[2]]

            # error handing for "ValueError: tile cannot extend outside image"
            if 0 in dst.shape:
                continue
            
            dst_path = f'{output_dir}dst_{splitext(basename(img_path))[0]}/{splitext(basename(img_path))[0]}_class{str(class_index).zfill(2)}_{str(instance_index).zfill(3)}.jpg'
            Image.fromarray(dst.astype(np.uint8)).save(dst_path)