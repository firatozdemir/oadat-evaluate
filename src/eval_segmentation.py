# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch
import os
import numpy as np
import h5py
import pickle
import decimal
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import tensorflow as tf
# import tensorflow_addons as tfa
print('TF version: ', tf.__version__)


# Import custom libs
import utils

def eval(model, task_str, datasets_parent_dir, fname_out):
    hpobj = utils.load_model_hyperparameters(task_str=task_str, datasets_parent_dir=datasets_parent_dir)

    fname = hpobj.fname_h5
    inds_test = hpobj.inds_test
    in_key = hpobj.in_key
    out_key = hpobj.out_key
    num_classes = 3
    labels = {'bg': 0, 'vessel': 1, 'skincurve': 2}

    precision_fn = lambda tp, fp: tp/(tp+fp)
    recall_fn = lambda tp, fn: tp/(tp+fn)
    f1_fn = lambda tp, fp, fn: 2*tp/(2*tp + fp + fn) # also dice
    IoU_fn = lambda tp, fp, fn: tp/(tp + fp + fn) # also Jaccard index

    @tf.function
    def scaleclip_batch_tf(x):
        x = x/tf.reduce_max(x, axis=(1,2), keepdims=True)
        x = tf.clip_by_value(x, clip_value_min=-0.2, clip_value_max=1.)
        return x

    bs = 100 ## batchsize
    num_samples = len(inds_test)
    d_res = {'precision': np.inf*np.ones((num_samples,num_classes)), 'recall': np.inf*np.ones((num_samples,num_classes)), 'dice': np.inf*np.ones((num_samples,num_classes)), 'iou': np.inf*np.ones((num_samples,num_classes)), 'TP': np.inf*np.ones((num_samples,num_classes)) , 'FP': np.inf*np.ones((num_samples,num_classes)), 'TN': np.inf*np.ones((num_samples,num_classes)), 'FN': np.inf*np.ones((num_samples,num_classes))}
    for i in range(0, num_samples, bs):
        if i+bs >= num_samples:
            iend = num_samples
        else:
            iend = i+bs
        with h5py.File(fname, 'r') as fh:
            ind_range = inds_test[i:iend]
            x = fh[in_key][ind_range,...] # [bs, 256, 256]
            ytrue = fh[out_key][ind_range,...] # [bs, 256, 256]
        x_t = tf.convert_to_tensor(x)
        ytrue_t = tf.convert_to_tensor(ytrue, dtype=tf.uint8)
        x_t = scaleclip_batch_tf(x_t)
        ypred_t = model(x_t, training=False)
        ypred_t = tf.nn.softmax(ypred_t, axis=-1) 
        
        x = np.squeeze(x_t.numpy())
        ytrue = np.squeeze(ytrue_t.numpy())    
        ypred = np.squeeze(ypred_t.numpy())
        ypred_lbl = np.argmax(ypred, axis=-1)
        
        for lbl in range(num_classes):
            predP = np.reshape(ypred_lbl == lbl, (ypred_lbl.shape[0], -1)) #shape: [bs, -1]
            trueP = np.reshape(ytrue == lbl, (ytrue.shape[0], -1))
            TP = np.logical_and(predP, trueP)
            FP = np.logical_and(predP, ~trueP)
            TN = np.logical_and(~predP, ~trueP)
            FN = np.logical_and(~predP, trueP)
            TP = np.sum(TP, axis=-1)
            FP = np.sum(FP, axis=-1)
            TN = np.sum(TN, axis=-1)
            FN = np.sum(FN, axis=-1)

            precision = precision_fn(tp=TP, fp=FP)
            recall = recall_fn(tp=TP, fn=FN)
            dice = f1_fn(tp=TP, fp=FP, fn=FN)
            IoU = IoU_fn(tp=TP, fp=FP, fn=FN)

            d_res['precision'][i:iend, lbl] = precision
            d_res['recall'][i:iend, lbl] = recall
            d_res['dice'][i:iend, lbl] = dice
            d_res['iou'][i:iend, lbl] = IoU
            d_res['TP'][i:iend, lbl] = TP
            d_res['FP'][i:iend, lbl] = FP
            d_res['TN'][i:iend, lbl] = TN
            d_res['FN'][i:iend, lbl] = FN
        logging.info('Completed %d/%d.' % (iend, num_samples))

    logging.info(f"Average metrics:\n class 0 (BG): {str([k+': '+str(np.mean(d_res[k][:,0])) for k in d_res.keys()])} \nclass 1 (vessel): {str([k+': '+str(np.mean(d_res[k][:,1])) for k in d_res.keys()])}\nclass 2 (skincurve): {str([k+': '+str(np.mean(d_res[k][:,2])) for k in d_res.keys()])}")

    metrics_to_report = ['dice', 'iou']
    for label_ind in [1,2]:
        print(f'Label: {list(labels.keys())[label_ind]}')
        str = ""
        for metric in metrics_to_report:
            v_mean = np.mean(d_res[metric][:,label_ind])
            v_std = np.std(d_res[metric][:,label_ind])
            v_std_dec = decimal.Decimal(v_std)
            str += f"{v_mean:.3f} ± {v_std_dec:.1e} \t\t"
        print(''.join([f"{m}\t" for m in metrics_to_report]))
        print(f'{str}\n')

    with open(fname_out, 'wb') as fh:
        pickle.dump(d_res, fh, pickle.HIGHEST_PROTOCOL)
    logging.info('Results saved to %s.' % fname_out)

def main(args):
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices([], 'GPU') # do not use any GPU
    tf.config.set_visible_devices(gpus[0], 'GPU') ## Use only GPU 0
    for gpu_instance in tf.config.list_physical_devices('GPU'): 
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    mpm_obj = utils.Manage_Pretrained_Models()
    ###############################################
    ## Modify this part according to your needs ###
    datasets_parent_dir = '/home/firat/docs/dlbirhoui/parsed_data2'
    task_str = 'swfd_lv128,sc'
    fname_out = os.path.join('/home/firat/saved_models/DLBIRHOUI/datasets_and_benchmarks/oadat_evaluate', task_str, 'eval.p')

    ## If evaluating one of the provided pretrained models use this
    model = mpm_obj.load_model(task_str=task_str)
    ## if evaluating another serialized model, use this
    # path_serialized_model = '/home/firat/saved_models/DLBIRHOUI/datasets_and_benchmarks/oadat_evaluate/swfd_lv128,li/serialized_model_step_140000'
    # model = tf.keras.models.load_model(path_serialized_model, compile=False)
    ###############################################


    eval(model, task_str, datasets_parent_dir, fname_out)
    
if __name__ == "__main__":
    main(0)