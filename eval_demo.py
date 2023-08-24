import tensorflow as tf
import sys, os
sys.path.append("src")
import utils
import eval_segmentation
import eval_translation
import parse_args
import tf_utils

def main():
    args = parse_args.parse()
    tf_utils.setup_gpu(gpu_id=args.gpu)
    mpm_obj = utils.Manage_Pretrained_Models()
    task_str = args.task_str

    if args.path_serialized_model is None: 
        model = mpm_obj.load_model(task_str=task_str)
    else:
        path_serialized_model = args.path_serialized_model
        model = tf.keras.models.load_model(path_serialized_model, compile=False)

    datasets_parent_dir = args.datasets_parent_dir
    fname_out = args.fname_out
    if not os.path.exists(os.path.dirname(fname_out)):
        os.makedirs(os.path.dirname(fname_out))

    if str(args.mode).lower() == 'translate' or str(args.mode).lower() == 'translation':
        eval_translation.eval(model, task_str, datasets_parent_dir, fname_out)
    elif str(args.mode).lower() == 'segment' or str(args.mode).lower() == 'segmentation':
        eval_segmentation.eval(model, task_str, datasets_parent_dir, fname_out)

if __name__ == "__main__":
    main()