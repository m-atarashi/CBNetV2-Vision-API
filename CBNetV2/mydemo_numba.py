import numpy as np
from PIL import Image
from numba import jit, uint8, float32, float64, void

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
    result = inference_detector(model, images)
    return [result[0][sepc_class_index], result[1][sepc_class_index]]


@jit(uint8[:,:,:](uint8[:,:,:], float32[:], uint8[:,:], float64), nopython=True)
def crop_instance(image, coord, mask, score_thr=0.3):
    # threshold check. coord[4] is the probabilty score
    if coord[4] <= score_thr:
        return np.ones((1,1,1), dtype=np.uint8)
        
    # convert bool to uint8 and reshape from 2D to 3D
    mask = mask.astype(np.uint8).reshape(*mask.shape, 1)
    masked_instance = image * mask

    # coord[:4] is BBox coordinate
    coord =coord.astype(np.uint16)[:4]
    cropped_instance = masked_instance[coord[1]:coord[3], coord[0]:coord[2]]

    # error handing for "ValueError: tile cannot extend outside image"
    if 0 in cropped_instance.shape:
        return np.ones((1,1,1), dtype=np.uint8)
    return cropped_instance


@jit(uint8[:,:,:,:](uint8[:,:,:], float32[:,:], uint8[:,:,:], float64), nopython=True)
def get_cropped_instances(image, coords, masks, score_thr=0.3):
    cropped_instances = np.array([crop_instance(image, coords[i], masks[i], score_thr) for i in range(len(coords))])
    return cropped_instances


def save_cropped_instances(cropped_instances, output_dir):
    for i, cropped_instance in enumerate(cropped_instances):
        if cropped_instance == np.ones((1,1,1), dtype=np.uint8):
            continue
        output_path = f'{output_dir}/book_{str(i).zfill(4)}.jpg'
        Image.fromarray(cropped_instance.astype(np.uint8)).save(dst_path)




# @jit(void(u1[:,:,:], f8[:,:], b1[:,:,:], f8, u1), nopython=True)
# def save_masked_image(image, coords, masks, score_thr=0.3, output_dir='../data/outputs/dst/'):
#     for i in range(len(coords)):
#         cropped_instance = crop_instance(image, coords[i], masks[i], score_thr)
#         if cropped_instance != np.ones((1,1,1), dtype=np.uint8):
#             dst_path = f'{output_dir}/book_{str(i).zfill(4)}.jpg'
#             Image.fromarray(cropped_instance.astype(np.uint8)).save(dst_path)