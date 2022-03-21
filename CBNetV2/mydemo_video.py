# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# from os.path import basename, splitext
# from glob import glob
# import ffmpeg


# CBNetV2_HOME = '/home/m-atarashi/CBNetV2/CBNetV2/'

# configs = [
#     f'{CBNetV2_HOME}configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py',
#     f'{CBNetV2_HOME}configs/cbnet/htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py'
# ]
# checkpoints = [
#     f'{CBNetV2_HOME}checkpoints/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth',
#     f'{CBNetV2_HOME}checkpoints/htc_cbv2_swin_large22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.pth'
# ]

# device = 'cuda:0'


# def demo(img, model, score_thr=0.3):
#     # inference the demo image
#     result = inference_detector(model, img)
#     output = '../demo_data/outputs/output_frames/' + splitext(basename(img))[0] + '_output.jpg'
#     model.show_result(img, result, score_thr=score_thr, out_file=output)


# def main():
#     model_index = 1
#     # init a detector
#     model = init_detector(configs[model_index], checkpoints[model_index], device=device)
    
#     imgs = sorted(glob('../demo_data/outputs/input_frames/*.png'))
#     for img in imgs:
#         demo(img, model)


# if __name__ == "__main__":
#     main()