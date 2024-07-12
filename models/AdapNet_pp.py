import tensorflow as tf
from tensorflow.keras import layers

# https://github.com/DeepSceneSeg/AdapNet-pp

# @article{valada2020self,
#   title={Self-supervised model adaptation for multimodal semantic segmentation},
#   author={Valada, Abhinav and Mohan, Rohit and Burgard, Wolfram},
#   journal={International Journal of Computer Vision},
#   volume={128},
#   number={5},
#   pages={1239--1285},
#   year={2020},
#   publisher={Springer}
# }

class AdapNet_pp:
    def __init__(self, num_classes=12, input_shape=(512, 512, 3)):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.eAspp_rate = [3, 6, 12]
        self.res_units = [3, 4, 6, 3]
        self.res_filters = [256, 512, 1024, 2048]
        self.res_strides = [1, 2, 2, 1]
        self.res_dilations = [1, 1, 1, 1]

    def _setup(self):
        ### RGB Model
        self.rgb_b0_out  = self._start_unit(self.rgb_input, name="rgb")
        #block1
        self.rgb_b1_out = self._resnet_unit_v2(self.rgb_b0_out , self.res_filters[0], self.res_strides[0], "rgb_conv2_block1", shortcut=False)
        for i in range(2, self.res_units[0]+1):
            self.rgb_b1_out = self._resnet_unit_v2(self.rgb_b1_out, self.res_filters[0], 1, "rgb_conv2_block%d"%i)
        #block2
        self.rgb_b2_out = self._resnet_unit_v2(self.rgb_b1_out, self.res_filters[1], self.res_strides[1], "rgb_conv3_block1", shortcut=False)
        for i in range(2, self.res_units[1]+1):
            self.rgb_b2_out = self._resnet_unit_v2(self.rgb_b2_out, self.res_filters[1], 1, "rgb_conv3_block%d"%i)
        #block3
        self.rgb_b3_out = self._resnet_unit_v2(self.rgb_b2_out, self.res_filters[2], self.res_strides[2], "rgb_conv4_block1", shortcut=False)
        for i in range(2, self.res_units[2]+1):
            self.rgb_b3_out = self._resnet_unit_v2(self.rgb_b3_out, self.res_filters[2], 1, "rgb_conv4_block%d"%i)
        #block4
        self.rgb_b4_out = self._resnet_unit_v2(self.rgb_b3_out, self.res_filters[3], self.res_strides[3], "rgb_conv5_block1", shortcut=False)
        for i in range(2, self.res_units[3]+1):
            self.rgb_b4_out = self._resnet_unit_v2(self.rgb_b4_out, self.res_filters[3], 1, "rgb_conv5_block%d"%i)
        ##skip
        self.rgb_skip1 = self._conv_batchN_relu(self.rgb_b1_out, 1, 1, 24, name="rgb_skip1", relu=False)
        self.rgb_skip2 = self._conv_batchN_relu(self.rgb_b2_out, 1, 1, 24, name="rgb_skip2", relu=False)
        ##eAspp
        self.rgb_eAspp_out = self._eASPP(self.rgb_b4_out, name="rgb_eASPP")

        ### Upsample/Decoder
        self.rgb_deconv_up1 = tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer="he_normal")(self.rgb_eAspp_out)
        self.rgb_deconv_up1 = tf.keras.layers.BatchNormalization()(self.rgb_deconv_up1)

        self.rgb_up1 = self._conv_batchN_relu(tf.concat([self.rgb_deconv_up1, self.rgb_skip2], -1), 3, 1, 256, name="rgb_up1a")
        self.rgb_up1 = self._conv_batchN_relu(self.rgb_up1, 3, 1, 256, name="rgb_up1b")

        self.rgb_deconv_up2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer="he_normal")(self.rgb_up1)
        self.rgb_deconv_up2 = tf.keras.layers.BatchNormalization()(self.rgb_deconv_up2)

        self.rgb_up2 = self._conv_batchN_relu(tf.concat([self.rgb_deconv_up2, self.rgb_skip1], -1), 3, 1, 256, name="rgb_up2a")
        self.rgb_up2 = self._conv_batchN_relu(self.rgb_up2, 3, 1, 256, name="rgb_up2b")
        self.rgb_up2 = self._conv_batchN_relu(self.rgb_up2, 1, 1, self.num_classes, name="rgb_up1c")

        self.rgb_deconv_up3 = tf.keras.layers.Conv2DTranspose(self.num_classes, kernel_size=8, strides=(4, 4), padding="same", kernel_initializer="he_normal")(self.rgb_up2)
        self.rgb_deconv_up3 = tf.keras.layers.BatchNormalization()(self.rgb_deconv_up3)

        self.rgb_softmax = tf.keras.layers.Softmax(axis=-1, name="rgb_softmax")(self.rgb_deconv_up3)
        self.outputs = self.rgb_softmax

    def _start_unit(self, x, name):
        outputs = tf.keras.layers.BatchNormalization(name='%s_conv1_bn'%name)(x)
        outputs = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', name='%s_conv1_conv'%name)(x)
        outputs = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="%s_pool1_pool"%name)(outputs)

        return outputs

    
    def _resnet_unit_v2(self, x, filters, stride=1, name=None, shortcut=True, kernel_size=3):
      # https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/resnet.py
        """A residual block for ResNet*_v2.

        Args:
            x: Input tensor.
            filters: No of filters in the bottleneck layer.
            kernel_size: Kernel size of the bottleneck layer. Defaults to `3`.
            stride: Stride of the first layer. Defaults to `1`.
            shortcut: Use convolution shortcut if `True`, otherwise
                use identity shortcut. Defaults to `True`
            name(optional): Name of the block

        Returns:
            Output tensor for the residual block.
        """
        if tf.keras.backend.image_data_format() == "channels_last":
            bn_axis = 3
        else:
            bn_axis = 1

        preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_preact_bn")(x)
        preact = layers.Activation("relu", name=name + "_preact_relu")(preact)

        if not shortcut:
            shortcut = layers.Conv2D(filters, 1, strides=stride, name=name + "_0_conv")(preact)
        else:
            shortcut = (layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x)

        x = layers.Conv2D(filters // 4, 1, strides=1, use_bias=False, name=name + "_1_conv")(preact)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x)
        x = layers.Activation("relu", name=name + "_1_relu")(x)

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
        x = layers.Conv2D(filters // 4, kernel_size, strides=stride, use_bias=False, name=name + "_2_conv",)(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(x)
        x = layers.Activation("relu", name=name + "_2_relu")(x)

        x = layers.Conv2D(filters, 1, name=name + "_3_conv")(x)
        x = layers.Add(name=name + "_out")([shortcut, x])
        return x


    def _conv_batchN_relu(self, x, kernel_size, stride, filters, name=None, relu=True):
        out = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding="same", name=name, kernel_initializer="he_normal")(x)
        out = tf.keras.layers.BatchNormalization(name="%s_bn"%name if name else None)(out)
        if relu:
            out = tf.keras.layers.ReLU(name="%s_relu"%name if name else None)(out)
        return out

    def _aconv_batchN_relu(self, x, kernel_size, dilation_rate, filters, name=None, relu=True):
        out = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, dilation_rate=dilation_rate, padding="same", name=name, kernel_initializer="he_normal")(x)
        out = tf.keras.layers.BatchNormalization(name="%s_bn"%name if name else None)(out)
        if relu:
            out = tf.keras.layers.ReLU(name="%s_relu"%name if name else None)(out)
        return out

    def _eASPP(self, x, name):
        IA = self._conv_batchN_relu(x, 1, 1, 256, name="%s_A"%name)

        IB = self._conv_batchN_relu(x, 1, 1, 64, name="%s_B1"%name)
        IB = self._aconv_batchN_relu(IB, 3, self.eAspp_rate[0], 64, name="%s_B2"%name)
        IB = self._aconv_batchN_relu(IB, 3, self.eAspp_rate[0], 64, name="%s_B3"%name)
        IB = self._conv_batchN_relu(IB, 1, 1, 256, name="%s_B4"%name)

        IC = self._conv_batchN_relu(x, 1, 1, 64, name="%s_C1"%name)
        IC = self._aconv_batchN_relu(IC, 3, self.eAspp_rate[1], 64, name="%s_C2"%name)
        IC = self._aconv_batchN_relu(IC, 3, self.eAspp_rate[1], 64, name="%s_C3"%name)
        IC = self._conv_batchN_relu(IC, 1, 1, 256, name="%s_C4"%name)

        ID = self._conv_batchN_relu(x, 1, 1, 64, name="%s_D1"%name)
        ID = self._aconv_batchN_relu(ID, 3, self.eAspp_rate[2], 64, name="%s_D2"%name)
        ID = self._aconv_batchN_relu(ID, 3, self.eAspp_rate[2], 64, name="%s_D3"%name)
        ID = self._conv_batchN_relu(ID, 1, 1, 256, name="%s_D4"%name)

        IE = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="%s_E1"%name)(x)
        IE = self._conv_batchN_relu(IE, 1, 1, 256, name="%s_E2"%name)
        IE = tf.keras.layers.UpSampling2D(size=(x.shape[1] // IE.shape[1], x.shape[2] // IE.shape[2]), interpolation="bilinear", name="%s_E3"%name)(IE)
        
        concat = tf.keras.layers.Concatenate(name="%s_add"%name, axis=-1)([IA, IB, IC, ID, IE])

        eAspp_out = self._conv_batchN_relu(concat, 1, 1, 256, name="%s_out"%name, relu=False)
        return eAspp_out

    def _load_pretrained(self, model):
        def remove_prefix(string):
            if string.startswith("rgb_"):
                return string[len("rgb_"):]
            elif string.startswith("depth_"):
                return string[len("depth_"):]
            else:
                return string

        weight_model = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_shape=(self.input_shape[0], self.input_shape[1], 3))

        for l in model.layers:
            layer_name = l.name
            if layer_name.startswith(("rgb_", "depth_")):
                layer_name_no_prefix = remove_prefix(layer_name)
                try:
                    if layer_name_no_prefix in [layer.name for layer in weight_model.layers]:
                        weight_layer = weight_model.get_layer(layer_name_no_prefix)
                        l.set_weights(weight_layer.get_weights())
                        # print(f"Loaded weights for layer: {layer_name} (mapped to {layer_name_no_prefix})")
                    # else:
                    #     print(f"No matching layer found in pre-trained model for layer: {layer_name}")
                except Exception as e:
                    print(f"Error loading weights for layer {layer_name}: {str(e)}")
            # else:
            #     print(f"Skipping layer: {layer_name} (not prefixed with 'rgb_' or 'depth_')")
        return model

    def __call__(self):
        self.rgb_input = tf.keras.layers.Input(shape=self.input_shape)
        self._setup()
        model = tf.keras.Model(inputs=self.rgb_input, outputs=self.outputs, name="AdapNet++")
        model = self._load_pretrained(model)
        return model
