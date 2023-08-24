import argparse

def parse():
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
    parser.add_argument(
        "--mode",
        type=str,
        help="Pick one of ['translate', 'segment']. Default: translate.",
        default="translate",
        required=False,
    )
    parser.add_argument(
        "--gpu", 
        type=int,
        help="ID of GPU to use. cpu: -1, otherwise will use cuda:gpu. Default:gpu:0",
        default=0,
        required=False,
    )

    args = parser.parse_args()
    return args