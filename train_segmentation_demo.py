import sys, os
sys.path.append("src")
import train_segmentation
datasets_parent_dir = 'data/OADAT-mini' # assuming datasets downloaded here.
task_str = 'seg_swfd_lv128,sc'
logdir = 'trained_models/seg_swfd_lv128,sc'
if not os.path.exists(logdir):
    os.makedirs(logdir)
args = train_segmentation.ExpSetup(datasets_parent_dir=datasets_parent_dir, task_str=task_str, logdir=logdir, batch_size=10, num_epochs_train=5)
fname_serialized_model = train_segmentation.train(args)
