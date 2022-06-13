# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch
import os
# os.environ['NUMEXPR_MAX_THREADS']='20'
import numpy as np
import time
import datetime
import copy
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import tensorflow as tf
# import tensorflow_addons as tfa
print('TF version: ', tf.__version__)


# Import custom libs
import architectures
import generators
import utils
import tf_utils

class ExpSetup:
    '''A simple experiment setup class to define hyper-parameters and train/val sets'''
    def __init__(self, datasets_parent_dir=None, task_str=None, logdir=None, load_specific_ckpt=None):
        ###############################
        ### Customize the following block
        self.datasets_parent_dir = '/home/firat/docs/dlbirhoui/parsed_data2' if datasets_parent_dir is None else datasets_parent_dir
        task_str = "seg_ss32,vc" if task_str is None else task_str
        self.logdir = os.path.join('/home/firat/saved_models/DLBIRHOUI/datasets_and_benchmarks/oadat_evaluate', task_str) if logdir is None else logdir
        ###############################
        self.hpobj = utils.ImageSegmentationSetups(task_str=task_str, datasets_parent_dir=self.datasets_parent_dir)
        self.prng = np.random.RandomState(42)
        # Training hyper-parameters
        self.num_epochs_train = 150
        self.freq_log = 500 # #steps for each summary export
        self.freq_eval_val = 5_000 # frequency to evaluate validation set [steps]
        self.batch_size = 25
        self.num_steps_train = int(self.num_epochs_train * len(self.hpobj.inds_train)//self.batch_size)
        self.imsize = [256, 256]  # original is [256, 256]
        self.fname_ckpt = os.path.join(self.logdir, 'model_weights.h5')
        self.fname_train_helper = os.path.join(self.logdir, 'train_helper.pkl')
        self.training_helper = utils.TrainingHelper(fname_save=self.fname_train_helper)

        # Dataset details.
        self.fname_h5 = self.hpobj.fname_h5
        self.in_key = self.hpobj.in_key
        self.out_key = self.hpobj.out_key
        self.labels = {'bg': 0, 'vessel': 1, 'skincurve': 2}
        self.use_inv_freq_labels = self.hpobj.use_inv_freq_labels
        self.label_inv_freq = self.hpobj.label_inv_freq 
        self.out_activation = None
        @tf.function
        def mapping_in_tf(x):
            x = x/tf.reduce_max(x)
            x = tf.clip_by_value(x, clip_value_min=-0.2, clip_value_max=1.)
            x = tf.expand_dims(x, axis=-1)
            return x
        self.mapping_in_tf = mapping_in_tf
        @tf.function
        def mapping_out_tf(x, depth=3):
            return tf.one_hot(x, depth=depth)
        self.mapping_out_tf = mapping_out_tf
        # self.mapping_in = lambda x: np.expand_dims(scaleclip_fn(x), axis=-1)
        self.mapping_in = None
        self.mapping_out = None
        
        self.data_preprocessing = 'scaleclip' # 'standardize', 'normalize', 'none', 'scaleclip'
        self.dataset_train, self.dataset_val = None, None
        self.setup_datasets() 
        self.out_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) 
        if self.use_inv_freq_labels:
            @tf.function
            def loss_fn(y_true, y_pred):
                class_weights = tf.constant(self.label_inv_freq, dtype=tf.float32)
                pixel_weight = tf.reduce_sum(y_true * tf.reshape(class_weights, [1,1,1,-1]), axis=-1, keepdims=False)
                return self.out_loss(y_true, y_pred, sample_weight=pixel_weight)
            self.loss_fn = loss_fn
        else:
            # self.loss_fn = lambda y_true, y_pred: self.out_loss(y_true, y_pred)
            self.loss_fn = self.out_loss


        # Architecture details: Attention UNet with batch norm and a single residual connection
        self.l1_reg = 0.01
        self.l2_reg = 0.01
        in_shape = list(self.imsize) + [1]
        self.model = architectures.modified_unet(in_shape=in_shape, out_activations=self.out_activation, out_channels=len(self.labels), l1reg=self.l1_reg, l2reg=self.l2_reg)
        self.model.summary()

        # Optimizer details: Adam optimizer with linear warmup and exponential decay
        
        self.lr_schedule = tf_utils.LRSchedule_LinearWarmup(warmup_steps=10_000., peak_learning_rate=1.e-4, decay_rate=0.98, decay_steps=1e3, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.95) # 
        self.global_gradient_clipnorm = 1e10

        # Setup checkpoint manager and try to restore model weights
        self.setup_checkpoint_manager(load_specific_ckpt=load_specific_ckpt)

    def setup_datasets(self):

        x_dtype, y_dtype = tf.float32, tf.uint8
        num_parallel_calls=tf.data.AUTOTUNE
        transforms = self.mapping_in
        transforms_target = self.mapping_out
        tf_mapping_in = self.mapping_in_tf
        tf_mapping_out = self.mapping_out_tf
        
        inds = self.hpobj.inds_train
        self.gen_train = generators.Generator_Paired_Input_Output(fname_h5=self.fname_h5, inds=inds, in_key=self.in_key, out_key=self.out_key, transforms=transforms, transforms_target=transforms_target, prng=self.prng, shuffle=True)
        self.training_helper.add_generator(name='gen_train', gen=self.gen_train)
        num_generator_threads_training = np.min((self.batch_size, 10)) # no point making #threads more than the minibatch size
        dataset_train = tf.data.Dataset.from_tensor_slices(['gen_train_%d'%id for id in range(num_generator_threads_training)])
        dataset_train = dataset_train.interleave(lambda x: tf.data.Dataset.from_generator(lambda: copy.deepcopy(self.gen_train), output_types=(x_dtype, y_dtype)), num_parallel_calls=num_parallel_calls, deterministic=False)
        dataset_train = dataset_train.map(lambda x, y: (tf_mapping_in(x), tf_mapping_out(y)), num_parallel_calls=num_parallel_calls,)
        dataset_train = dataset_train.batch(self.batch_size, drop_remainder=True, deterministic=False, num_parallel_calls=num_parallel_calls,)
        self.dataset_train = dataset_train.prefetch(buffer_size=num_parallel_calls)

        inds = self.hpobj.inds_val
        self.gen_val = generators.Generator_Paired_Input_Output(fname_h5=self.fname_h5, inds=inds, in_key=self.in_key, out_key=self.out_key, transforms=transforms, transforms_target=transforms_target, prng=self.prng, shuffle=True)
        self.training_helper.add_generator(name='gen_val', gen=self.gen_val)
        num_generator_threads_val = np.min((self.batch_size, 3)) # no point making #threads more than the minibatch size
        dataset_val = tf.data.Dataset.from_tensor_slices(['gen_val_%d'%id for id in range(num_generator_threads_val)])
        dataset_val = dataset_val.interleave(lambda x: tf.data.Dataset.from_generator(lambda: copy.deepcopy(self.gen_val), output_types=(x_dtype, y_dtype)), num_parallel_calls=num_parallel_calls, deterministic=False)
        dataset_val = dataset_val.map(lambda x, y: (tf_mapping_in(x), tf_mapping_out(y)), num_parallel_calls=num_parallel_calls,)
        dataset_val = dataset_val.batch(self.batch_size, drop_remainder=True, deterministic=False, num_parallel_calls=num_parallel_calls,)
        self.dataset_val = dataset_val.prefetch(buffer_size=num_parallel_calls)

    def setup_checkpoint_manager(self, load_specific_ckpt=None):
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.save_manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=self.logdir, max_to_keep=3, \
             checkpoint_name='model_ckpt')
        self.save_manager_best_val_loss = tf.train.CheckpointManager(checkpoint=ckpt, directory=self.logdir, max_to_keep=3, 
            checkpoint_name='model_best_val_loss_ckpt')

        if load_specific_ckpt is None:
            ckpt.restore(self.save_manager.latest_checkpoint)
            ckptname = self.save_manager.latest_checkpoint
        else:
            try:
                ckptname = utils.get_ckptname(logdir=self.logdir, id=load_specific_ckpt)  # assume basename for ckpt manager is passed. Find the latest one matching this.
            except:
                ckptname = load_specific_ckpt  # assume full ckpt path is passed.
            ckpt.restore(ckptname).assert_existing_objects_matched().expect_partial()
        if ckptname:
            logging.info(f"Restored from {ckptname}")
            self.last_loaded_ckpt = ckptname
            self.training_helper.load()
            self.num_epoch = (self.training_helper.epoch_current)  # get the epoch you left off
            self.best_val_loss = self.training_helper.best_val_loss
        else:
            logging.info("Initializing training from scratch.")
            self.num_epoch = 0
            self.best_val_loss = np.inf

    def save(self, epoch_current, best_val_loss, ckpt_number=None):
        self.save_manager.save(checkpoint_number=ckpt_number)
        self.training_helper.save(epoch_current, best_val_loss)
        

def train(args):

    ## Read a few important variables
    logdir = args.logdir
    model = args.model
    save = args.save
    labels = args.labels
    optimizer = args.optimizer
    loss_fn = args.loss_fn
    num_epochs_train = args.num_epochs_train
    num_steps_train = args.num_steps_train
    num_epoch = args.num_epoch
    best_val_loss = args.best_val_loss
    freq_log = args.freq_log
    freq_eval_val = args.freq_eval_val
    save_manager_best_val_loss = args.save_manager_best_val_loss
    dataset_train = args.dataset_train
    dataset_val = args.dataset_val
    global_gradient_clipnorm = args.global_gradient_clipnorm
    logging.info('Logdir: %s, Training will be %d epochs (%d steps).'% (logdir, num_epochs_train, num_steps_train))
    practical_epoch_fn = lambda step: step/(len(args.hpobj.inds_train)//args.batch_size)
    

    ## Setup summaries
    summary_writer_train = tf.summary.create_file_writer(logdir=os.path.join(logdir, 'train'))
    summary_writer_val = tf.summary.create_file_writer(logdir=os.path.join(logdir, 'val')) 
    
    ## evaluating validation set 
    def eval_and_log(model, dataset, loss_fn, log=True, writer=None, num_step=None):
        '''Function to evaluate a dataset and log (optional) avg results'''
        avg_loss = tf.keras.metrics.Mean(name='loss_', dtype=tf.float32)
        soft_dice_fn = tf_utils.soft_dice
        d_dice = {}
        for x, y in dataset:
            y_pred = model(x, training=False)
            loss = loss_fn(y_true=y, y_pred=y_pred)
            avg_loss.update_state(loss)
            dice_score = soft_dice_fn(logits=y_pred, labels=y, num_labels=len(labels))
            for i, k in enumerate(labels.keys()):
                if 'Dice_ch_%d'%(i+1) not in d_dice:
                    d_dice['Dice_ch_%d'%(i+1)] = tf.keras.metrics.Mean(name='dice_eval', dtype=tf.float32)
                d_dice['Dice_ch_%d'%(i+1)].update_state(dice_score[i])
        if writer is not None:
            assert (num_step is not None) # num_step must be provided.
            d_scalars = {'loss': avg_loss.result()}
            d_scalars['dice'] = tf.reduce_mean(dice_score)
            tf_utils.export_summary_scalars(dict_name_and_val=d_scalars, step=num_step, writer=writer)
            d_images = {}
            d_images['x_ch_1'] = tf_utils.im_norm(x[0:1])
            for k in labels.keys():
                label = labels[k]
                d_images['y_true_ch_%s'%label] = y[0:1, ..., label:label+1]
                d_images['y_pred_ch_%s'%label] = tf_utils.im_norm(y_pred[0:1, ..., label:label+1])
            tf_utils.export_summary_images(dict_name_and_val=d_images, step=num_step, writer=writer)
            d_annotation_maps = {}
            d_annotation_maps['y_true_map'] = tf.expand_dims(tf.argmax(y, axis=-1)[0:1], axis=-1)
            d_annotation_maps['y_pred_map'] = tf.expand_dims(tf.argmax(tf.nn.softmax(y_pred, axis=-1)[0:1], axis=-1), axis=-1)
            tf_utils.export_summary_label_maps(dict_name_and_val=d_annotation_maps, step=num_step, writer=writer, num_labels=len(labels))
        return d_scalars

    ### Trace model graph for tensorboard
    @tf.function
    def model_graph(model, x):
        return model(x, training=False)

    tf.summary.trace_on(graph=True, profiler=False)
    x,_ = next(iter(dataset_train))
    _ = model_graph(model, x)
    with summary_writer_train.as_default():
        tf.summary.trace_export(name='model_trace', step=0, profiler_outdir=args.logdir)
        logging.info('exported model graph')

    ## Training Loop
    avg_loss_epoch = tf.keras.metrics.Mean(name='loss', dtype=tf.float32) # variable will keep track of average of loss value within the epoch
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32) # variable will keep track of average of loss value since last freq_log
    d_dice = {}
    saved_file = False 
    profiler_scheduler = tf_utils.ProfilerScheduler(init_delay_steps=1001, freq_run_epochs=100, profiler_run_duration=50)
    profiler_options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3, python_tracer_level=1, device_tracer_level=1, delay_ms=None)
    for epoch in range(num_epoch, num_epochs_train):
        t_epoch_start = time.time()
        for x,y in dataset_train: # loop over training step for a full epoch
            if profiler_scheduler.check_to_stop(step=optimizer.iterations.numpy()):
                tf.profiler.experimental.stop()
                logging.info("Profiler stopped!")
            loss, y_pred = tf_utils.train_step(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn, global_gradient_clipnorm=global_gradient_clipnorm)
            num_step = optimizer.iterations
            if profiler_scheduler.check_to_start(step=optimizer.iterations.numpy(), epoch=epoch):
                logging.info("Profiler is starting!")
                tf.profiler.experimental.start(args.logdir, options=profiler_options)
            avg_loss.update_state(loss) # aggregate values since last flush
            avg_loss_epoch.update_state(loss)
            dice_score = tf_utils.soft_dice(logits=y_pred, labels=y, num_labels=len(labels))
            for i, k in enumerate(labels.keys()):
                if 'Dice_ch_%d'%(i+1) not in d_dice:
                    d_dice['Dice_ch_%d'%(i+1)] = tf.keras.metrics.Mean(name='dice_training', dtype=tf.float32)
                d_dice['Dice_ch_%d'%(i+1)].update_state(dice_score[i])
            if tf.equal(optimizer.iterations % freq_log, 0):
                d_scalars = {
                    'loss': avg_loss.result(), 
                    'lr': args.lr_schedule(num_step),
                    'dice': tf.reduce_mean(dice_score),
                    }
                tf_utils.export_summary_scalars(dict_name_and_val=d_scalars, step=optimizer.iterations, writer=summary_writer_train)
                d_images = {}
                d_images['x_ch_1'] = tf_utils.im_norm(x[0:1])
                for k in labels.keys():
                    label = labels[k]
                    d_images['y_true_ch_%s'%label] = y[0:1, ..., label:label+1]
                    d_images['y_pred_ch_%s'%label] = tf_utils.im_norm(y_pred[0:1, ..., label:label+1])
                tf_utils.export_summary_images(dict_name_and_val=d_images, step=optimizer.iterations, writer=summary_writer_train)
                str_format_scalars = ', '.join([': '.join([k, '%.5f'%d_scalars[k]]) for k in d_scalars.keys()])
                logging.info('Step %d: Training set: %s' % (num_step, str_format_scalars))
                avg_loss.reset_states() #reset kept history of loss

            if tf.equal(optimizer.iterations % freq_eval_val, 0):
                d_scalars = eval_and_log(model=model, dataset=dataset_val, loss_fn=loss_fn, log=True, writer=summary_writer_val, num_step=optimizer.iterations)
                loss_val = d_scalars['loss']
                if loss_val < best_val_loss:
                    logging.info('New best avg validation loss (%.3f), saving.' % loss_val)
                    best_val_loss = loss_val
                    save_manager_best_val_loss.save(checkpoint_number=optimizer.iterations)
                str_format_scalars = ', '.join([': '.join([k, '%.3f'%d_scalars[k]]) for k in d_scalars.keys()])
                logging.info('Step %d: Validation set: %s' % (num_step, str_format_scalars))

            if not saved_file:
                args.training_helper.save_training_script(file=__file__, logdir=logdir)
                saved_file = True

        epoch_real = practical_epoch_fn(step=num_step.numpy())
        t_epoch_duration = time.time() - t_epoch_start
        str_epoch_duration = str(datetime.timedelta(seconds=t_epoch_duration))
        logging.info('Epoch %d (practical epoch: %.1f, step %d): Average training loss: %.3f. Avg. epoch time: %s.' % (epoch, epoch_real, num_step.numpy(), avg_loss_epoch.result(), str_epoch_duration))
        avg_loss_epoch.reset_states()
        ## Regular save of the model.
        save(epoch_current=epoch, best_val_loss=best_val_loss, ckpt_number=optimizer.iterations)
        if epoch_real >= num_epochs_train:
            break

    logging.info('Training is completed.')


def main(args):
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices([], 'GPU') # do not use any GPU
    tf.config.set_visible_devices(gpus[1], 'GPU') ## Use only GPU 0
    for gpu_instance in tf.config.list_physical_devices('GPU'): 
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    args = ExpSetup()
    train(args)
    
if __name__ == "__main__":
    main(0)