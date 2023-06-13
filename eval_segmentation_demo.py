import tensorflow as tf
import sys
sys.path.append("src")
import utils
import eval_segmentation

mpm_obj = utils.Manage_Pretrained_Models()
task_str = 'seg_swfd_lv128,sc'

## Comment below if model is readily trained within this session
model = mpm_obj.load_model(task_str=task_str)
## Otherwise place path to the serialized model below
# path_serialized_model = 'dir/to/serialized/model'
# model = tf.keras.models.load_model(path_serialized_model, compile=False)
##

datasets_parent_dir = 'data/OADAT-mini' # assuming datasets downloaded here.
fname_out = 'trained_models/seg_swfd_lv128,sc/eval.p'
if not os.path.exists(logdir):
    os.makedirs(logdir)

eval_segmentation.eval(model, task_str, datasets_parent_dir, fname_out)
