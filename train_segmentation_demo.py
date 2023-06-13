import sys, os
sys.path.append("src")
import train_segmentation
import argparse

def main(args):
    args = parse(args)

    datasets_parent_dir = args.datasets_parent_dir
    task_str = args.task_str
    logdir = args.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    args = train_segmentation.ExpSetup(datasets_parent_dir=datasets_parent_dir, task_str=task_str, logdir=logdir, batch_size=args.batch_size, num_epochs_train=args.num_epochs_train)
    fname_serialized_model = train_segmentation.train(args)

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
        "-l",
        "--logdir",
        type=str,
        help="logging dir for tensorboard events and model ckpts",
        default="trained_models/seg_swfd_lv128,sc",
        required=False,
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        help="minibatch size",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--num_epochs_train", 
        type=int,
        help="# epochs to train",
        default=5,
        required=False,
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(0)