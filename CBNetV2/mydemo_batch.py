import numpy as np
from PIL import Image

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


CBNetV2_HOME = '/home/m-atarashi/CBNetV2/CBNetV2/'
device = 'cuda:0'

configs = [
    f'{CBNetV2_HOME}configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py',
    f'{CBNetV2_HOME}configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py'
]
checkpoints = [
    f'{CBNetV2_HOME}checkpoints/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth',
    f'{CBNetV2_HOME}checkpoints/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth'
]

COCO_classes = {0: 'person', 73: 'book'}


def load_model(config_file=configs[1], checkpoint_file=checkpoints[1], device=device):
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    return model


def inference(images, model, score_thr=0.3):
    # inference images batch
    results = inference_detector(model, images)
    return results


def save_instances(image, result, score_thr=0.3, output_dir='../data/outputs/dst/'):
    # class 0: person, class 73: book
    for class_index in [73]:
        # null check, result[1] is segmentation data
        if not result[1][class_index]:
            continue
        
        for instance_index, insetance_mask in enumerate(result[1][class_index]):
            # threshold check. result[0][class_index][instance_index][4] is the probabilty score
            if result[0][class_index][instance_index][4] < score_thr:
                continue

            mask = insetance_mask.astype(np.uint8)
            if mask.shape[:2] != image.shape[:2]:
                # enlarge the mask size
                mask = np.array(Image.fromarray(mask).resize((image.shape[1], image.shape[0]), Image.BICUBIC))
            mask = mask.reshape(*mask.shape, 1)

            dst = image * mask
            scale = mask.shape[0]/insetance_mask.astype(np.uint8).shape[0]
            # result[0][class_index][instance_index][0:4] is BBox coordinate
            coord = (scale*np.array(result[0][class_index][instance_index])).astype(np.uint16)[:4]
            dst = dst[coord[1]:coord[3], coord[0]:coord[2]]

            # error handing for "ValueError: tile cannot extend outside image"
            if 0 in dst.shape:
                continue
            
            dst_path = f'{output_dir}/{COCO_classes[class_index]}_{str(instance_index).zfill(4)}.jpg'
            Image.fromarray(dst.astype(np.uint8)).save(dst_path)