import tensorflow as tf

# https://github.com/TUI-NICR/ESANet/tree/main

# @inproceedings{seichter2021efficient,
#   title={Efficient rgb-d semantic segmentation for indoor scene analysis},
#   author={Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael},
#   booktitle={2021 IEEE international conference on robotics and automation (ICRA)},
#   pages={13525--13531},
#   year={2021},
#   organization={IEEE}
# }

class ESANet():
    def __init__(self, num_classes=16, input_shape=(512,512,4)):
        self.input_size = input_shape
        self.inputs = tf.keras.Input(shape=self.input_size)
        
        self.context_module =  ContextModule('ppm', 2048, 512)

        self.dec_module1 = Decoder_module(out_channels=512)
        self.dec_module2 = Decoder_module(out_channels=256)
        self.dec_module3 = Decoder_module(out_channels=128)

        self.conv_1 = tf.keras.Sequential([
                  tf.keras.layers.Conv2D(512, kernel_size=1, padding='same', kernel_initializer='he_normal'),
                  tf.keras.layers.BatchNormalization(),
                  tf.keras.layers.ReLU(),
              ])
        self.conv_2 = tf.keras.Sequential([
                  tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal'),
                  tf.keras.layers.BatchNormalization(),
                  tf.keras.layers.ReLU(),
              ])
        self.conv_3 = tf.keras.Sequential([
                  tf.keras.layers.Conv2D(128, kernel_size=1, padding='same', kernel_initializer='he_normal'),
                  tf.keras.layers.BatchNormalization(),
                  tf.keras.layers.ReLU(),
              ])

        self.learned_up1 = LearnedUp()
        self.learned_up2 = LearnedUp()
        # Classifier
        self.classifier = tf.keras.layers.Conv2D(num_classes, kernel_size=3, padding='same', kernel_initializer='he_normal')

    def __call__(self):
        # ResNet50
        blocks = [3, 4, 6, 3]
        resnet_strides = [1, 2, 2, 2]
        resnet_dilations = [1, 1, 1, 1]

        # Encoder
        rgb_input = self.inputs[:,:,:,:3]
        depth_input = tf.expand_dims(self.inputs[:,:,:,3], axis=-1)

        ################### Depth Branch ##############################
        #starter_block
        block0_depth = self._start_block(depth_input, name='depth')
        depth_maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="depth_pool1_pool")(block0_depth)
        #block1
        block1_depth = self._bottleneck_resblock(depth_maxpool, 256, resnet_strides[0], resnet_dilations[0], 'depth_conv2_block1', identity_connection=False)
        for i in range(2, blocks[0]+1):
            block1_depth = self._bottleneck_resblock(block1_depth, 256, 1, 1, 'depth_conv2_block%d'%i)
        #block2
        block2_depth = self._bottleneck_resblock(block1_depth, 512, resnet_strides[1], resnet_dilations[1], 'depth_conv3_block1', identity_connection=False)
        for i in range(2, blocks[1]+1):
            block2_depth = self._bottleneck_resblock(block2_depth, 512, 1, 1, 'depth_conv3_block%d'%i)
        #block3
        block3_depth = self._bottleneck_resblock(block2_depth, 1024, resnet_strides[2], resnet_dilations[2], 'depth_conv4_block1', identity_connection=False)
        for i in range(2, blocks[2]+1):
            block3_depth = self._bottleneck_resblock(block3_depth, 1024, 1, 1, 'depth_conv4_block%d'%i)
        #block4
        block4_depth = self._bottleneck_resblock(block3_depth, 2048, resnet_strides[3], resnet_dilations[3], 'depth_conv5_block1', identity_connection=False)
        for i in range(2, blocks[3]+1):
            block4_depth = self._bottleneck_resblock(block4_depth, 2048, 1, 1, 'depth_conv5_block%d'%i)

        ################### RGB Branch ##############################
        #starter_block
        block0_rgb = self._start_block(rgb_input, name='rgb')
        #block1
        block0_fused = SqueezeAndExciteFusionAdd(C=64)(block0_rgb, block0_depth)
        rgb_maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="rgb_pool1_pool")(block0_fused)
        block1_rgb = self._bottleneck_resblock(rgb_maxpool, 256, resnet_strides[0], resnet_dilations[0], 'rgb_conv2_block1', identity_connection=False)
        for i in range(2, blocks[0]+1):
            block1_rgb = self._bottleneck_resblock(block1_rgb, 256, 1, 1, 'rgb_conv2_block%d'%i)
        #block2
        block1_fused = SqueezeAndExciteFusionAdd(C=256)(block1_rgb, block1_depth)
        block2_rgb = self._bottleneck_resblock(block1_fused, 512, resnet_strides[1], resnet_dilations[1], 'rgb_conv3_block1', identity_connection=False)
        for i in range(2, blocks[1]+1):
            block2_rgb = self._bottleneck_resblock(block2_rgb, 512, 1, 1, 'rgb_conv3_block%d'%i)
        #block3
        block2_fused = SqueezeAndExciteFusionAdd(C=512)(block2_rgb, block2_depth)
        block3_rgb = self._bottleneck_resblock(block2_fused, 1024, resnet_strides[2], resnet_dilations[2], 'rgb_conv4_block1', identity_connection=False)
        for i in range(2, blocks[2]+1):
            block3_rgb = self._bottleneck_resblock(block3_rgb, 1024, 1, 1, 'rgb_conv4_block%d'%i)
        #block4
        block3_fused = SqueezeAndExciteFusionAdd(C=1024)(block3_rgb, block3_depth)
        block4_rgb = self._bottleneck_resblock(block3_fused, 2048, resnet_strides[3], resnet_dilations[3], 'rgb_conv5_block1', identity_connection=False)
        for i in range(2, blocks[3]+1):
            block4_rgb = self._bottleneck_resblock(block4_rgb, 2048, 1, 1, 'rgb_conv5_block%d'%i)
        block4_fused = SqueezeAndExciteFusionAdd(C=2048)(block4_rgb, block4_depth)

        ################### Context Module ##############################
        context_output = self.context_module(block4_fused)

        ################### Decoder Branch ##############################
        encoder_features1 = self.conv_1(block3_fused)
        encoder_features2 = self.conv_2(block2_fused)
        encoder_features3 = self.conv_3(block1_fused)

        decoder_features1 = self.dec_module1(context_output, encoder_features1)
        decoder_features2 = self.dec_module2(decoder_features1, encoder_features2)
        decoder_features3 = self.dec_module3(decoder_features2, encoder_features3)

        last_conv = self.classifier(decoder_features3)
        output = self.learned_up1(last_conv)
        output = self.learned_up2(output)

        model = tf.keras.Model(inputs=self.inputs, outputs=output, name='ESANet')
        model = self._load_pretrained(model)
        return model

    def _start_block(self, x, name):
        outputs = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', name='%s_conv1_conv'%name)(x)
        outputs = tf.keras.layers.BatchNormalization(name='%s_conv1_bn'%name)(outputs)
        outputs = tf.keras.layers.Activation("relu", name='%s_conv1_relu'%name)(outputs)
        # outputs = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="%s_pool1_pool"%name)(outputs)
        return outputs

    def _bottleneck_resblock(self, x, filters, stride, dilation_factor, name, identity_connection=True):
        assert filters % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name='%s_0_conv'%name)(x)
            o_b1 = tf.keras.layers.BatchNormalization(name='%s_0_bn'%name)(o_b1)
        else:
            o_b1 = x
        # branch2
        o_b2a = tf.keras.layers.Conv2D(filters / 4, kernel_size=1, strides=1, padding='same', name='%s_1_conv'%name)(x)
        o_b2a = tf.keras.layers.BatchNormalization(name='%s_1_bn'%name)(o_b2a)
        o_b2a = tf.keras.layers.Activation("relu", name='%s_1_relu'%name)(o_b2a)

        o_b2b = tf.keras.layers.Conv2D(filters / 4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name='%s_2_conv'%name)(o_b2a)
        o_b2b = tf.keras.layers.BatchNormalization(name='%s_2_bn'%name)(o_b2b)
        o_b2b = tf.keras.layers.Activation("relu", name='%s_2_relu'%name)(o_b2b)

        o_b2c = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name='%s_3_conv'%name)(o_b2b)
        o_b2c = tf.keras.layers.BatchNormalization(name='%s_3_bn'%name)(o_b2c)

        # add
        outputs = tf.keras.layers.Add(name='%s_add'%name)([o_b1, o_b2c])
        # relu
        outputs = tf.keras.layers.Activation("relu", name='%s_out'%name)(outputs)
        return outputs

    def _load_pretrained(self, model):
        def remove_prefix(string):
            if string.startswith("rgb_"):
                return string[len("rgb_"):]
            elif string.startswith("depth_"):
                return string[len("depth_"):]
            else:
                return string

        weight_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(self.input_size[0], self.input_size[1], 3))

        for l in model.layers:
            layer_name = l.name
            if layer_name.startswith(("rgb_", "depth_")):
                layer_name_no_prefix = remove_prefix(layer_name)
                try:
                    if layer_name_no_prefix in [layer.name for layer in weight_model.layers]:
                        weight_layer = weight_model.get_layer(layer_name_no_prefix)
                        l.set_weights(weight_layer.get_weights())
                        # print(f"Loaded weights for layer: {layer_name} (mapped to {layer_name_no_prefix})")
                    else:
                        print(f"No matching layer found in pre-trained model for layer: {layer_name}")
                except Exception as e:
                    print(f"Error loading weights for layer {layer_name}: {str(e)}")
            # else:
            #     print(f"Skipping layer: {layer_name} (not prefixed with 'rgb_' or 'depth_')")
        return model



class ContextModule(tf.keras.layers.Layer):
    def __init__(self, context_module_name, channels_in, channels_out,
                      activation=tf.keras.layers.ReLU(), upsampling_mode='bilinear'):
        super(ContextModule, self).__init__()

        self.context_module = tf.identity
        if 'ppm' in context_module_name:
            if context_module_name == 'ppm-1-2-4-8':
                bins = (1, 2, 4, 8)
            else:
                bins = (1, 5)
            self.context_module = PyramidPoolingModule(
                channels_in, channels_out,
                bins=bins,
                activation=activation,
                upsampling_mode=upsampling_mode)
        else :
              raise NotImplementedError("%s Not Implemented."%context_module_name)

    def __call__(self, x):
        return self.context_module(x)

class AdaptiveAvgPool2D(tf.keras.layers.Layer):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2D, self).__init__()
        self.output_size = output_size

    def __call__(self, inputs):
        height = inputs.shape[1]
        width = inputs.shape[2]

        target_height = self.output_size[0]
        target_width = self.output_size[1]
        
        stride_height = height // target_height
        stride_width = width // target_width
        # Calculate pooling size for height and width
        pool_size_height = height - (target_height - 1) * stride_height
        pool_size_width = width - (target_width - 1) * stride_width
        # Perform average pooling
        pooled_output = tf.nn.avg_pool(inputs, ksize=[1, pool_size_height, pool_size_width, 1],
                                       strides=[1, stride_height, stride_width, 1], padding='VALID')
        return pooled_output

class PyramidPoolingModule(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, bins=(1, 2, 3, 6),
                 activation=tf.keras.layers.ReLU(), upsampling_mode='bilinear'):
        super(PyramidPoolingModule, self).__init__()

        reduction_dim = in_dim // len(bins)
        self.features = []
        self.upsampling_mode = upsampling_mode
        for bin_size in bins:
            self.features.append(tf.keras.Sequential([
                # tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
                AdaptiveAvgPool2D(output_size=(bin_size, bin_size)),
                tf.keras.layers.Conv2D(reduction_dim, kernel_size=1, padding='same', kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization(),
                activation
            ]))
            
        self.final_conv = tf.keras.Sequential([
                  tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding='same', kernel_initializer='he_normal'),
                  tf.keras.layers.BatchNormalization(),
                  activation,
              ])

    def __call__(self, x):
        x_size = x.shape
        out = [x]
        for f in self.features:
            y = f(x)
            if self.upsampling_mode == 'nearest':
                out.append(tf.image.resize(y, (x_size[1], x_size[2]), method='nearest'))
            elif self.upsampling_mode == 'bilinear':
                out.append(tf.image.resize(y, (x_size[1], x_size[2]), method='bilinear'))
            else:
                raise NotImplementedError(
                    'For the PyramidPoolingModule only nearest and bilinear '
                    'interpolation are supported. '
                    f'Got: {self.upsampling_mode}'
                )
        out = tf.concat(out, axis=-1)
        out = self.final_conv(out)
        return out


class NonBottleneck1D(tf.keras.layers.Layer):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    # expansion = 1

    def __init__(self, out_channels, stride=1, downsample=None, groups=1, base_width=None, dilation=1, norm_layer=None,
                 activation=tf.keras.layers.ReLU(), residual_only=False):
        super(NonBottleneck1D, self).__init__()
        # warnings.warn('parameters groups, base_width and norm_layer are ignored in NonBottleneck1D')
        self.conv3x1_1 = tf.keras.layers.Conv2D(out_channels, (3, 1), strides=(stride, 1), padding='same', use_bias=True, kernel_initializer='he_normal')
        self.conv1x3_1 = tf.keras.layers.Conv2D(out_channels, (1, 3), strides=(1, stride), padding='same', use_bias=True, kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1e-03)
        self.act = activation
        self.conv3x1_2 = tf.keras.layers.Conv2D(out_channels, (3, 1), padding='same', use_bias=True, dilation_rate=(dilation, 1), kernel_initializer='he_normal')
        self.conv1x3_2 = tf.keras.layers.Conv2D(out_channels, (1, 3), padding='same', use_bias=True, dilation_rate=(1, dilation), kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1e-03)
        self.dropout_rate = 0.5
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def __call__(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout_rate != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        return self.act(output + identity)

class CustomInitializer(tf.keras.initializers.Initializer):
    def __init__(self, kernel_values):
        self.kernel_values = kernel_values

    def __call__(self, shape, dtype=None):
        if not shape == self.kernel_values.shape:
            raise ValueError("Shape of kernel_values does not match the shape of the kernel.")
        return tf.constant(self.kernel_values, dtype=dtype)

class LearnedUp(tf.keras.layers.Layer):
    def __init__(self, factor=2):
        super(LearnedUp, self).__init__()
        self.nearest_up = tf.keras.layers.UpSampling2D(size=(factor,factor), interpolation='nearest')

        # Define the kernel values that mimics bilinear interpolation
        kernel_values = [[0.0625, 0.1250, 0.0625],
                        [0.1250, 0.2500, 0.1250],
                        [0.0625, 0.1250, 0.0625]]
        # Create the custom initializer with the kernel values
        initializer = CustomInitializer(kernel_values)
        # Define the convolutional layer with specific kernel values (kernel_size=3)
        self.depth_wise = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', kernel_initializer=initializer)

    def __call__(self, x):
        x = self.nearest_up(x)
        out = self.depth_wise(x)
        return out

class Decoder_module(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(Decoder_module, self).__init__()
        self.conv_3x3 = tf.keras.Sequential([
                  tf.keras.layers.Conv2D(out_channels, kernel_size=1, padding='same', kernel_initializer='he_normal'),
                  tf.keras.layers.BatchNormalization(),
                  tf.keras.layers.ReLU(),
              ])
        blocks = []
        for i in range(3) :
            blocks.append(NonBottleneck1D(out_channels=out_channels))
        self.decoder_blocks = tf.keras.Sequential(blocks)
        self.learned_up = LearnedUp()
        self.add = tf.keras.layers.Add()

    def __call__(self, decoder_features, encoder_features):
        out = self.conv_3x3(decoder_features)
        out = self.decoder_blocks(out)
        out = self.learned_up(out)
        out = self.add([out, encoder_features])
        return out

class SqueezeAndExciteFusionAdd(tf.keras.layers.Layer):
    def __init__(self, C, reduction=16):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Conv2D(C // reduction, kernel_size=1, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(C, kernel_size=1, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('sigmoid')
        ])

        self.se_depth = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Conv2D(C // reduction, kernel_size=1, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(C, kernel_size=1, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation('sigmoid')
        ])

    def __call__(self, rgb, depth):
        rgb_weight = self.se_rgb(rgb)
        rgb = tf.multiply(rgb, rgb_weight)
        
        depth_weight = self.se_depth(depth)
        depth = tf.multiply(depth, depth_weight)
        
        out = tf.add(rgb, depth)
        return out

