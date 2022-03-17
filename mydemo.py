import argparse
import enum
import numpy as np
from ast import parse
from os import makedirs
from os.path import basename, splitext
from PIL import Image

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


configs = [
    'configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py',
    'configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py'
]
checkpoints = [
    'checkpoints/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth',
    'checkpoints/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth'
]

device = 'cuda:0'


def save_masked_image(img_path, result, score_thr=0.3):
    makedirs(f'data/outputs/dst_{splitext(basename(img_path))[0]}', exist_ok=True)
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
            
            dst_path = f'data/outputs/dst_{splitext(basename(img_path))[0]}/{splitext(basename(img_path))[0]}_class{str(class_index).zfill(2)}_{str(instance_index).zfill(3)}.jpg'
            Image.fromarray(dst.astype(np.uint8)).save(dst_path)


def demo(img_path, config_file, checkpoint_file, score_thr=0.3):
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    result = inference_detector(model, img_path)
    # show the result image
    show_result_pyplot(model, img_path, result, score_thr=score_thr)
    # save the result image
    output = f'data/outputs/{splitext(basename(img_path))[0]}_output.jpg'
    model.show_result(img_path, result, score_thr=score_thr, out_file=output)

    return result


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint_index", type=int, help="index of checkpoint")
    parser.add_argument("-i", "--img_path", type=str, help="path of an image which the model inference")
    parser.add_argument("-i_origin", "--original_img_path", type=str, help="path of the original image with overlapping bounding box.The scale ratio must be equal to the input image.")
    parser.add_argument("--score_thr", type=float, default=0.3, help="threshold socre to determines the bounding boxes to be displayed (defalt is 0.3)")
    parser.add_argument("-l", "--list", action="store_true", help="list information about available checkpoints")
    args = parser.parse_args()

    if args.list:
        print("\nList available checkpoints:")
        for i, c in enumerate(checkpoints):
            print(f"index: {i} | path: {basename(c)}")
        print("\n")

    if args.checkpoint_index is None or args.img_path is None:
        return

    if len(checkpoints) < args.checkpoint_index:
        print(f"There are only {len(checkpoints)} available chechkpoints. Please confirm available checkpoints to use -c or --checkpoint option")
        return

    return args


def main():
    args = parser()
    if args is None:
        return args

    checkpoint_index  = args.checkpoint_index
    img_path          = args.img_path
    original_img_path = args.original_img_path if args.original_img_path else args.img_path
    score_thr         = args.score_thr

    result = demo(img_path, configs[checkpoint_index], checkpoints[checkpoint_index], score_thr)
    save_masked_image(original_img_path, result, score_thr)


if __name__ == "__main__":
    main()