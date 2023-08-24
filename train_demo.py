import sys, os
sys.path.append("src")
import train_segmentation
import train_translation
import parse_args
import tf_utils

def main():
    args = parse_args.parse()
    tf_utils.setup_gpu(gpu_id=args.gpu)
    datasets_parent_dir = args.datasets_parent_dir
    task_str = args.task_str
    logdir = args.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if str(args.mode).lower() == 'translate' or str(args.mode).lower() == 'translation':
        args = train_translation.ExpSetup(datasets_parent_dir=datasets_parent_dir, task_str=task_str, logdir=logdir, batch_size=args.batch_size, num_epochs_train=args.num_epochs_train)
        fname_serialized_model = train_translation.train(args)
    elif str(args.mode).lower() == 'segment' or str(args.mode).lower() == 'segmentation':
        args = train_segmentation.ExpSetup(datasets_parent_dir=datasets_parent_dir, task_str=task_str, logdir=logdir, batch_size=args.batch_size, num_epochs_train=args.num_epochs_train)
        fname_serialized_model = train_segmentation.train(args)

if __name__ == "__main__":
    main()