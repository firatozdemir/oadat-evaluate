#!/bin/bash
bash download-oadat-mini.sh
python train_segmentation_demo.py --task_str="seg_swfd_lv128,sc" --datasets_parent_dir="data/OADAT-mini" --logdir="trained_models/seg_swfd_lv128,sc"
python eval_segmentation_demo.py --task_str="seg_swfd_lv128,sc" --datasets_parent_dir="data/OADAT-mini" --fname_out="trained_models/seg_swfd_lv128,sc/eval.p"