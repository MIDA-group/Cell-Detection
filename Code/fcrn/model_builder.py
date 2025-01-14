from tensorflow.keras import layers, regularizers, optimizers, Model, backend

# Set image data format to channels last (default in TF 2.x)
backend.set_image_data_format('channels_last')

# Weight decay constant
weight_decay = 1e-5

def _conv_bn_relu(nb_filter, kernel_size=(3,3), strides=(1,1)):
    def f(input_tensor):
        # Updated Convolution2D to Conv2D
        conv_a = layers.Conv2D(
            filters=nb_filter, 
            kernel_size=kernel_size, 
            strides=strides,
            kernel_initializer='orthogonal', 
            padding='same', 
            use_bias=False
        )(input_tensor)
        
        # Updated BatchNormalization
        norm_a = layers.BatchNormalization()(conv_a)
        act_a = layers.Activation('relu')(norm_a)
        return act_a
    return f
    
def _conv_bn_relu_x2(nb_filter, kernel_size=(3,3), strides=(1,1)):
    def f(input_tensor):
        # First convolution with regularization
        conv_a = layers.Conv2D(
            filters=nb_filter, 
            kernel_size=kernel_size, 
            strides=strides,
            kernel_initializer='orthogonal', 
            padding='same', 
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            bias_regularizer=regularizers.l2(weight_decay)
        )(input_tensor)
        
        norm_a = layers.BatchNormalization()(conv_a)
        act_a = layers.Activation('relu')(norm_a)
        
        # Second convolution
        conv_b = layers.Conv2D(
            filters=nb_filter, 
            kernel_size=kernel_size, 
            strides=strides,
            kernel_initializer='orthogonal', 
            padding='same', 
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            bias_regularizer=regularizers.l2(weight_decay)
        )(act_a)
        
        norm_b = layers.BatchNormalization()(conv_b)
        act_b = layers.Activation('relu')(norm_b)
        return act_b
    return f

def FCRN_A_base(input_tensor):
    block1 = _conv_bn_relu(32)(input_tensor)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(block1)
    
    block2 = _conv_bn_relu(64)(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(block2)
    
    block3 = _conv_bn_relu(128)(pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(block3)
    
    block4 = _conv_bn_relu(512)(pool3)
    
    up5 = layers.UpSampling2D(size=(2, 2))(block4)
    block5 = _conv_bn_relu(128)(up5)
    
    up6 = layers.UpSampling2D(size=(2, 2))(block5)
    block6 = _conv_bn_relu(64)(up6)
    
    up7 = layers.UpSampling2D(size=(2, 2))(block6)
    block7 = _conv_bn_relu(32)(up7)
    return block7

def FCRN_A_base_v2(input_tensor):
    block1 = _conv_bn_relu_x2(32)(input_tensor)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(block1)
    
    block2 = _conv_bn_relu_x2(64)(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(block2)
    
    block3 = _conv_bn_relu_x2(128)(pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(block3)
    
    block4 = _conv_bn_relu(512)(pool3)
    
    up5 = layers.UpSampling2D(size=(2, 2))(block4)
    block5 = _conv_bn_relu_x2(128)(up5)
    
    up6 = layers.UpSampling2D(size=(2, 2))(block5)
    block6 = _conv_bn_relu_x2(64)(up6)
    
    up7 = layers.UpSampling2D(size=(2, 2))(block6)
    block7 = _conv_bn_relu_x2(32)(up7)
    return block7

def U_net_base(input_tensor, nb_filter=64):
    block1 = _conv_bn_relu_x2(nb_filter)(input_tensor)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(block1)
    
    block2 = _conv_bn_relu_x2(nb_filter)(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(block2)
    
    block3 = _conv_bn_relu_x2(nb_filter)(pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(block3)
    
    block4 = _conv_bn_relu_x2(nb_filter)(pool3)
    up4 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(block4), block3])
    
    block5 = _conv_bn_relu_x2(nb_filter)(up4)
    up5 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(block5), block2])
    
    block6 = _conv_bn_relu_x2(nb_filter)(up5)
    up6 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(block6), block1])
    
    block7 = _conv_bn_relu(nb_filter)(up6)
    return block7

def buildModel_FCRN_A(input_dim):
    input_ = layers.Input(shape=input_dim)
    act_ = FCRN_A_base(input_)
    
    density_pred = layers.Conv2D(
        filters=1, 
        kernel_size=(1, 1), 
        use_bias=False, 
        activation='linear',
        kernel_initializer='orthogonal', 
        padding='same', 
        name='pred'
    )(act_)
    
    model = Model(inputs=input_, outputs=density_pred)
    
    # Updated optimizer initialization
    opt = optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='mse')
    return model

def buildModel_FCRN_A_v2(input_dim):
    input_ = layers.Input(shape=input_dim)
    act_ = FCRN_A_base_v2(input_)
    
    density_pred = layers.Conv2D(
        filters=1, 
        kernel_size=(1, 1), 
        use_bias=False, 
        activation='linear',
        kernel_initializer='orthogonal', 
        padding='same', 
        name='pred'
    )(act_)
    
    model = Model(inputs=input_, outputs=density_pred)
    
    opt = optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='mse')
    return model

def buildModel_U_net(input_dim):
    input_ = layers.Input(shape=input_dim)
    act_ = U_net_base(input_, nb_filter=64)
    
    density_pred = layers.Conv2D(
        filters=1, 
        kernel_size=(1, 1), 
        use_bias=False, 
        activation='linear',
        kernel_initializer='orthogonal', 
        padding='same', 
        name='pred'
    )(act_)
    
    model = Model(inputs=input_, outputs=density_pred)
    
    # Changed to RMSprop from keras.optimizers
    opt = optimizers.RMSprop(learning_rate=1e-3)
    model.compile(optimizer=opt, loss='mse')
    return model