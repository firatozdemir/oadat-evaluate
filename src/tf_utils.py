# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch
import tensorflow as tf

@tf.function
def export_summary_scalars(dict_name_and_val, step, writer):
    if not isinstance(dict_name_and_val, dict):
        raise AssertionError('dict_name_and_val must be a dictionary.')
    with writer.as_default():
        for k in dict_name_and_val.keys():
            tf.summary.scalar(k, dict_name_and_val[k], step=step)

@tf.function
def export_summary_images(dict_name_and_val, step, writer):
    if not isinstance(dict_name_and_val, dict):
        raise AssertionError('dict_name_and_val must be a dictionary.')
    with writer.as_default():
        for k in dict_name_and_val.keys():
            tf.summary.image(k, dict_name_and_val[k], step=step)

@tf.function
def export_summary_label_maps(dict_name_and_val, step, writer, num_labels):
    '''Expects inputs to have integer values, each corresponding to a class label'''
    if not isinstance(dict_name_and_val, dict):
        raise AssertionError('dict_name_and_val must be a dictionary.')
    with writer.as_default():
        for k in dict_name_and_val.keys():
            im = dict_name_and_val[k]
            im = im / (num_labels-1)
            im = tf.cast(im*255, dtype=tf.uint8)
            tf.summary.image(k, im, step=step)

@tf.function
def im_norm(x):
    '''Scale and shift image to [0, 1]'''
    x -= tf.math.reduce_min(x) 
    return x/tf.math.reduce_max(x) 


@tf.function
def ssim_ms_fn(y_true, y_pred, filter_size_ssim, max_val=None):
    if max_val is None: #normalize input to range [0, 1]
        y_true -= tf.math.reduce_min(y_true) 
        y_true /= tf.math.reduce_max(y_true) 
        y_pred -= tf.math.reduce_min(y_pred) 
        y_pred /= tf.math.reduce_max(y_pred) 
        max_val = 1.0
    return tf.image.ssim_multiscale(y_true, y_pred, max_val=max_val, filter_size=filter_size_ssim, filter_sigma=1.5, k1=0.01, k2=0.03)
@tf.function
def ssim_fn(y_true, y_pred, filter_size_ssim, max_val=None):
    if max_val is None: #normalize input to range [0, 1]
        y_true -= tf.math.reduce_min(y_true) 
        y_true /= tf.math.reduce_max(y_true) 
        y_pred -= tf.math.reduce_min(y_pred) 
        y_pred /= tf.math.reduce_max(y_pred) 
        max_val = 1.0
    return tf.image.ssim(y_true, y_pred, max_val=max_val, filter_size=filter_size_ssim, filter_sigma=1.5, k1=0.01, k2=0.03)

@tf.function
def soft_dice(logits, labels, num_labels, epsilon=1.e-8):
    '''Expects logits and labels in a categorical form (one-hot encoded)'''
    flat_logits = tf.cast(tf.reshape(logits, [-1, num_labels]), tf.float32)
    flat_logits = tf.nn.softmax(flat_logits, axis=1) 
    flat_labels = tf.cast(tf.reshape(labels, [-1, num_labels]), dtype=tf.float32)
    intersection = tf.math.reduce_sum(tf.math.multiply(flat_logits, flat_labels), axis=0) # |X \cap Y| per class
    x = tf.math.reduce_sum(flat_logits, axis=0) #|X|
    y = tf.math.reduce_sum(flat_labels, axis=0) #|Y|
    dice_array = 2 * intersection / (x + y + epsilon)
    return dice_array

## Setup training step
@tf.function
def train_step(model, x, y, optimizer, loss_fn, global_gradient_clipnorm=1e10):
    '''Single training step'''
    with tf.GradientTape(persistent=False) as tape: #persistent=True if .gradient() will be called multiple times (e.g., multiple losses)
        y_pred = model(x, training=True)
        loss = loss_fn(y_true=y, y_pred=y_pred)
        gradients = tape.gradient(target=loss, sources=model.trainable_variables)
        if global_gradient_clipnorm is not None: # clip gradients if they are too large 
            gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clipnorm, name='clip_gradients_by_global_norm')
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, y_pred

class LRSchedule_LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, peak_learning_rate, decay_rate, decay_steps, staircase=False, init_lr=1e-5):
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        self.peak_learning_rate = peak_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
        self.staircase = staircase
        self.init_lr = init_lr

    @tf.function
    def decayed_learning_rate(self, step):
        if self.staircase:
            return self.peak_learning_rate * self.decay_rate ** (step // self.decay_steps)
        else:
            return self.peak_learning_rate * self.decay_rate ** (step / self.decay_steps)

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        if step <= self.warmup_steps:
            return self.init_lr + (self.peak_learning_rate-self.init_lr) * step / self.warmup_steps 
        else:
            return self.decayed_learning_rate(step-self.warmup_steps)

class ProfilerScheduler:
    '''A scheduler class for handling Tensorflow profiling'''
    def __init__(self, init_delay_steps, freq_run_epochs, profiler_run_duration=50):
        self.init_delay_steps = init_delay_steps
        self.freq_run_epochs = freq_run_epochs
        self.first_step = None
        self.profiler_run_duration = profiler_run_duration
        self.is_profiler_running = False
        self.profiler_start_step = None
        self.l_epochs_prof_run = []
    def check_to_start(self, step, epoch):
        if self.first_step is None: # first check will verify the initial step of the new/continuing training
            self.first_step = step
            return False
        if step - self.first_step < self.init_delay_steps: # don't start the profiler in the first N steps of a new/continuing training
            return False
        if epoch % self.freq_run_epochs == 0:
            if epoch in self.l_epochs_prof_run:
                return False
            if not self.is_profiler_running:
                self.is_profiler_running = True
                self.profiler_start_step = step
                self.l_epochs_prof_run.append(epoch)
                return True
        return False
    def check_to_stop(self, step):
        if self.is_profiler_running:
            if step - self.profiler_start_step == self.profiler_run_duration:
                self.is_profiler_running = False
                return True
        return False