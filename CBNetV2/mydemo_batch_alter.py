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

# class 0: person, class 73: book
COCO_classes = {0: 'person', 73: 'book'}
sepc_class_index = 73


def load_model(config_file=configs[1], checkpoint_file=checkpoints[1], device=device):
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    return model


def inference(images, model, score_thr=0.3):
    # inference images batch
    results = inference_detector(model, images)
    return results[:][0][sepc_class_index], results[:][1][sepc_class_index]


def save_instances(image, coords, masks, score_thr=0.3, output_dir):
    for i in range(len(coords))]
        # threshold check. coord[4] is the probabilty score
        if coords[4] <= score_thr:
            continue
            
        # convert bool to uint8 and reshape from 2D to 3D
        masks = masks.astype(np.uint8).reshape(*masks.shape, 1)
        masked_instance = image * masks

        # coord[:4] is BBox coordinate
        coords = coords.astype(np.uint16)[:4]
        cropped_instance = masked_instance[coords[1]:coords[3], coords[0]:coords[2]]

        # error handing for "ValueError: tile cannot extend outside image"
        if 0 in cropped_instance.shape:
            continue
        output_path = f'{output_dir}/book_{str(i).zfill(4)}.jpg'
        Image.fromarray(cropped_instance.astype(np.uint8)).save(dst_path)