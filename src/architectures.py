# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch

import tensorflow as tf

class AttentionGate(tf.keras.layers.Layer):
    '''Based on https://arxiv.org/abs/1804.03999, but for 2D. https://github.com/ozan-oktay/Attention-Gated-Networks'''
    def __init__(self, **kwargs):
        self.kernel_initializer = kwargs.pop('kernel_initializer', 'glorot_uniform')
        self.l1l2_reg = kwargs.pop('l1l2_reg', None)
        super().__init__(**kwargs)
                
    def build(self, input_shape):  # Create the state of the layer (weights)
        x_shape = input_shape[0]
        g_shape = input_shape[1]
        ndims_int = x_shape.as_list()[-1]
        self.x_ds = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='match_x_to_g_spatially')
        self.wx = tf.keras.layers.Conv2D(filters=ndims_int, kernel_size=1, activation=None, use_bias=False, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.l1l2_reg, name='Wx')
        self.wg = tf.keras.layers.Conv2D(filters=ndims_int, kernel_size=1, activation=None, use_bias=True, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.l1l2_reg, name='Wg')
        # self.relu_sum = tf.keras.layers.Lambda(lambda x:tf.keras.activations.relu(x[0]+x[1]), name='relu_on_sum')
        self.q_att = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None, use_bias=True, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.l1l2_reg, name='Phi')
        self.bilin_up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='bilinear_upsample')
        self.mul = tf.keras.layers.Multiply()
    
    def call(self, inputs):  # Defines the computation from inputs to outputs
        x = inputs[0]
        g = inputs[1]
        xl = x
        x = self.x_ds(x) #downsample
        x = self.wx(x)
        g = self.wg(g)
        x = tf.keras.activations.relu(x+g)
        # x = self.relu_sum((x,g))
        x = self.q_att(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.bilin_up(x)
        x = self.mul([xl, x])
        return x

class Residual_Conv2D_BN_Block(tf.keras.layers.Layer):
    '''Residual 2D Conv kernel with Batch Normalization.'''
    def __init__(self, filters, kernel_size, activation=None, strides=None, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            self.strides = (1,1)
        else:
            self.strides = strides
        self.filters =  filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'padding': self.padding,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
        })
        return config

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation=None, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1, name='BN1')
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation=None, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1, name='BN2')
        self.act = None if self.activation is None else tf.keras.layers.Activation(self.activation)
        self.conv_match = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(1,1), strides=self.strides, activation=None, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, name='match')
        self.sum = tf.keras.layers.Add(name='sum')

    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        if self.act is not None:
            x = self.act(x)
        x = self.conv2(x)
        if inputs.shape[-1] != x.shape[-1]:
            x_match = self.conv_match(inputs)
        else:
            x_match = inputs
        x = self.sum((x_match, x))
        x = self.bn2(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x

def modified_unet(in_shape, out_activations=None, out_channels=1, l1reg=None, l2reg=None):
    '''
    Example: unet_attention_model(in_shape=[256, 256, 1], out_channels=2) will produce output of shape [256, 256, 2]. 
    '''
    
    if (l1reg is None or l1reg == 0.) and (l2reg is None or l2reg == 0.):
        l1l2_reg = None
    else:
        l1l2_reg = tf.keras.regularizers.L1L2(l1=l1reg, l2=l2reg)            

    inputs = tf.keras.layers.Input(shape=in_shape, name='input')
    x = inputs

    conv_kernel_init = tf.keras.initializers.he_normal()
    conv_block = lambda filters, name=None: Residual_Conv2D_BN_Block(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=conv_kernel_init, kernel_regularizer=l1l2_reg, name=name)
    upsampling_fn = lambda name=None: tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=name)
    # lvl 1
    x = conv_block(filters=32, name='c1')(x)
    x_lvl1 = x
    # lvl 2
    x = tf.keras.layers.MaxPool2D((2,2), name='p1')(x)
    x = conv_block(filters=64, name='c2')(x)
    x_lvl2 = x
    # lvl 3
    x = tf.keras.layers.MaxPool2D((2,2), name='p2')(x)
    x = conv_block(filters=128, name='c3')(x)
    x_lvl3 = x
    # lvl 4
    x = tf.keras.layers.MaxPool2D((2,2), name='p3')(x)
    x = conv_block(filters=256, name='c4')(x)
    x_lvl4 = x
    # lvl 5
    x = tf.keras.layers.MaxPool2D((2,2), name='p4')(x)
    x = conv_block(filters=512, name='c5')(x)
    # lvl 4
    x_ag = AttentionGate(l1l2_reg=l1l2_reg, name='AG_4')([x_lvl4, x])
    x = upsampling_fn(name='up5')(x)
    x = tf.keras.layers.Concatenate(axis=-1, name='cat_lvl4')([x_ag, x])
    x = conv_block(filters=256, name='dc4')(x)
    # lvl 3
    x_ag = AttentionGate(l1l2_reg=l1l2_reg, name='AG_3')([x_lvl3, x])
    x = upsampling_fn(name='up4')(x)
    x = tf.keras.layers.Concatenate(axis=-1, name='cat_lvl3')([x_ag, x])
    x = conv_block(filters=128, name='dc3')(x)
    # lvl 2
    x_ag = AttentionGate(l1l2_reg=l1l2_reg, name='AG_2')([x_lvl2, x])
    x = upsampling_fn(name='up3')(x)
    x = tf.keras.layers.Concatenate(axis=-1, name='cat_lvl2')([x_ag, x])
    x = conv_block(filters=64, name='dc2')(x)
    # lvl 1
    x_ag = AttentionGate(l1l2_reg=l1l2_reg, name='AG_1')([x_lvl1, x])
    x = upsampling_fn(name='up2')(x)
    x = tf.keras.layers.Concatenate(axis=-1, name='cat_lvl1')([x_ag, x])
    x = conv_block(filters=32, name='dc1')(x)
    # output lvl
    x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1,1), activation=None, kernel_regularizer=l1l2_reg, name='preds')(x)
    if out_activations is not None:
        x = tf.keras.layers.Activation(out_activations)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name='modified_unet')
    return model