#!/bin/bash

python ../PaddleOCR/tools/infer/predict_system.py \
--image_dir="../data/inputs/book_0020.jpg" \
--det_model_dir="../PaddleOCR/checkpoints/ch_ppocr_server_v2.0_det_infer" \
--cls_model_dir="../PaddleOCR/checkpoints/ch_ppocr_mobile_v2.0_cls_infer" \
--rec_model_dir="../PaddleOCR/checkpoints/japan_mobile_v2.0_rec_infer" \
--use_angle_cls=true \
--rec_char_dict_path="../PaddleOCR/ppocr/utils/dict/japan_dict.txt" \
--vis_font_path="../PaddleOCR/doc/fonts/japan.ttc" \
--use_gpu=True \
--drop_score=0.5