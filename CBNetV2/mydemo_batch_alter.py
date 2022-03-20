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


def inference(images, model):
    # inference images batch
    results = inference_detector(model, images)
    results = np.array(results)
    return results[:, 0, sepc_class_index], results[:, 1, sepc_class_index]


def save_instances(image, coords, score, masks, score_thr=0.3, output_dir=f'{CBNetV2_HOME}../outputs/'):
    num_instances = len(coords)
    for i in range(num_instances):
        # threshold check. coords[i][4] is the probabilty score
        if coords[i][4] <= score_thr:
            continue
            
        masked_instance  = image * mask[i]
        cropped_instance = masked_instance[coords[i][1]:coords[i][3], coords[i][0]:coords[i][2]]

        # error handing for "ValueError: tile cannot extend outside image"
        if 0 in cropped_instance.shape:
            continue

        output_path = f'{output_dir}{COCO_classes[sepc_class_index]}_{str(i).zfill(4)}.jpg'
        Image.fromarray(cropped_instance.astype(np.uint8)).save(output_path)