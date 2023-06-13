import tensorflow as tf
import sys, os
sys.path.append("src")
import utils
import eval_segmentation
import argparse


def main(args):
    args = parse(args)

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

    eval_segmentation.eval(model, task_str, datasets_parent_dir, fname_out)

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task_str",
        type=str,
        help="target string",
        default='seg_swfd_lv128,sc',
        required=False,
    )
    parser.add_argument(
        "-d",
        "--datasets_parent_dir",
        type=str,
        help="path to dataset",
        default='data/OADAT-mini',
        required=False,
    )
    parser.add_argument(
        "-o",
        "--fname_out",
        type=str,
        help="filename for pickled output metrics",
        default="trained_models/seg_swfd_lv128,sc/eval.p",
        required=False,
    )
    parser.add_argument(
        "--path_serialized_model",
        type=str,
        help="path to serialized model",
        default=None, ## 'dir/to/serialized/model'
        required=False,
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(0)