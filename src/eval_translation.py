# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch
import os
import numpy as np
import h5py
import pickle
import copy
import sklearn.metrics
import skimage.metrics
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
    
    mae_fn = lambda ypred, ytrue: sklearn.metrics.mean_absolute_error(ytrue, ypred)
    rmse_fn = lambda ypred, ytrue: np.sqrt(skimage.metrics.mean_squared_error(ytrue, ypred))
    ssim_fn = lambda ypred, ytrue: skimage.metrics.structural_similarity(ytrue, ypred, data_range=ypred.max()-ypred.min())
    psnr_fn = lambda ypred, ytrue: skimage.metrics.peak_signal_noise_ratio(ytrue, ypred, data_range=ypred.max()-ypred.min())
    metrics = ['MAE', 'RMSE', 'SSIM', 'PSNR']

    @tf.function
    def scaleclip_batch_tf(x):
        x = x/tf.reduce_max(x, axis=(1,2), keepdims=True)
        x = tf.clip_by_value(x, clip_value_min=-0.2, clip_value_max=1.)
        return x

    def calc_std(sum_x, sum_x2, n):
        var = sum_x2/n - ((sum_x/n)**2)
        return np.sqrt(var)

    if 'msfd' in task_str:

        wavelengths = [700, 730, 760, 780, 800, 850]
        num_wavelengths = len(wavelengths)
        bs = 100 ## batchsize
        num_samples = len(inds_test)
        d_res_all = {}
        for i_w, wavelength in enumerate(wavelengths):
            in_key_curr = in_key[i_w]
            out_key_curr = out_key[i_w]
            if str(wavelength) not in in_key_curr or str(wavelength) not in out_key_curr:
                raise AssertionError('wavelength mismatch: wavelength: %d vs in_key: %s, out_key: %s' % (wavelength, in_key_curr, out_key_curr)) 
            d_res = {'MAE': np.inf*np.ones((num_samples,)), 'RMSE': np.inf*np.ones((num_samples,)), 'SSIM': np.inf*np.ones((num_samples,)), 'PSNR': np.inf*np.ones((num_samples,))}
            for i in range(0, num_samples, bs):
                if i+bs >= num_samples:
                    iend = num_samples
                else:
                    iend = i+bs
                with h5py.File(fname, 'r') as fh:
                    ind_range = inds_test[i:iend]
                    x = fh[in_key_curr][ind_range,...] # [bs, 256, 256]
                    ytrue = fh[out_key_curr][ind_range,...] # [bs, 256, 256]
                x_t = tf.convert_to_tensor(x)
                ytrue_t = tf.convert_to_tensor(ytrue)
                x_t = scaleclip_batch_tf(x_t)
                ytrue_t = scaleclip_batch_tf(ytrue_t)
                ypred_t = model(x_t, training=False)
                # ypred_t = scaleclip_batch_tf(ypred_t) ## apply scaleclip at prediction output to ensure intensity value range.ypred_t = scaleclip_batch_tf(ypred_t) ## apply scaleclip at prediction output to ensure intensity value range.
                
                x = np.squeeze(x_t.numpy())
                ytrue = np.squeeze(ytrue_t.numpy())    
                ypred = np.squeeze(ypred_t.numpy())
                
                for i_arr, i_metrics in enumerate(range(i, iend)):
                    d_res['MAE'][i_metrics] = mae_fn(ypred[i_arr,...], ytrue[i_arr,...])
                    d_res['RMSE'][i_metrics] = rmse_fn(ypred[i_arr,...], ytrue[i_arr,...])
                    d_res['SSIM'][i_metrics] = ssim_fn(ypred[i_arr,...], ytrue[i_arr,...])
                    d_res['PSNR'][i_metrics] = psnr_fn(ypred[i_arr,...], ytrue[i_arr,...])
                logging.info('Wavelength: %d (%d/%d). Completed %d/%d.' % (wavelength, (i_w+1), num_wavelengths, iend, num_samples))

            # logging.info(f"Wavelength: {wavelength} ({i_w+1}/{num_wavelengths}). Average metrics: {str([k+': '+str(np.mean(d_res[k])) for k in d_res.keys()])}")
            d_res_all[wavelength] = copy.deepcopy(d_res)

        str_ = ""
        for metric in metrics:
            sum_x = 0
            sum_x2 = 0
            l_v = []
            n = 0
            for wavelength in wavelengths:
                v_mean = np.mean(d_res_all[wavelength][metric])
                v_std = np.std(d_res_all[wavelength][metric])
                l_v += d_res_all[wavelength][metric].tolist()
                sum_x += np.sum(d_res_all[wavelength][metric])
                sum_x2 += np.sum(d_res_all[wavelength][metric]**2)
                n += len(d_res_all[wavelength][metric])
            overall_mean = sum_x / n
            overall_std = calc_std(sum_x, sum_x2, n)
            if metric == 'PSNR':
                str_ += f"{np.mean(overall_mean):.3f} ± {overall_std:.2f} \t"
            else:
                overall_std_dec = decimal.Decimal(overall_std)
                str_ += f"{np.mean(overall_mean):.3f} ± {overall_std_dec:.1e} \t"
        print(''.join([f"{m}\t" for m in metrics]))
        print(f"{str_}")
    else: #SWFD or SCD
        bs = 100 ## batchsize
        num_samples = len(inds_test)
        d_res = {'MAE': np.inf*np.ones((num_samples,)), 'RMSE': np.inf*np.ones((num_samples,)), 'SSIM': np.inf*np.ones((num_samples,)), 'PSNR': np.inf*np.ones((num_samples,))}
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
            ytrue_t = tf.convert_to_tensor(ytrue)
            x_t = scaleclip_batch_tf(x_t)
            ytrue_t = scaleclip_batch_tf(ytrue_t)
            ypred_t = model(x_t, training=False)
            # ypred_t = scaleclip_batch_tf(ypred_t) ## apply scaleclip at prediction output to ensure intensity value range.
            
            x = np.squeeze(x_t.numpy())
            ytrue = np.squeeze(ytrue_t.numpy())    
            ypred = np.squeeze(ypred_t.numpy())
            
            for i_arr, i_metrics in enumerate(range(i, iend)):
                d_res['MAE'][i_metrics] = mae_fn(ypred[i_arr,...], ytrue[i_arr,...])
                d_res['RMSE'][i_metrics] = rmse_fn(ypred[i_arr,...], ytrue[i_arr,...])
                d_res['SSIM'][i_metrics] = ssim_fn(ypred[i_arr,...], ytrue[i_arr,...])
                d_res['PSNR'][i_metrics] = psnr_fn(ypred[i_arr,...], ytrue[i_arr,...])
            logging.info('Completed %d/%d.' % (iend, num_samples))

        # logging.info(f"Average metrics: {str([k+': '+str(np.mean(d_res[k])) for k in d_res.keys()])}")

        str = ""
        for metric in metrics:
            v_mean = np.mean(d_res[metric])
            v_std = np.std(d_res[metric])
            v_std_dec = decimal.Decimal(v_std)
            if metric == 'PSNR':
                str += f"{v_mean:.3f} ± {v_std:.2f} \t"
            else:
                str += f"{v_mean:.3f} ± {v_std_dec:.1e} \t"
        print(''.join([f"{m}\t" for m in metrics]))
        print(str)

    with open(fname_out, 'wb') as fh:
        pickle.dump(d_res, fh, pickle.HIGHEST_PROTOCOL)
    logging.info('Results saved to %s.' % fname_out)

def main(args):
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices([], 'GPU') # do not use any GPU
    tf.config.set_visible_devices(gpus[1], 'GPU') ## Use only GPU 0
    for gpu_instance in tf.config.list_physical_devices('GPU'): 
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    mpm_obj = utils.Manage_Pretrained_Models()
    ###############################################
    ## Modify this part according to your needs ###
    datasets_parent_dir = '/home/firat/docs/dlbirhoui/parsed_data2'
    task_str = 'swfd_lv128,li'
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