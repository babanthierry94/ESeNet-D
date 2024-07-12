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

class AdapNet_pp():
    """
    ResNet-50
    output_stride fixed 16
    """
    def __init__(self, num_classes=12, input_shape=(512, 512, 4)):
        super(AdapNet_pp, self).__init__()
        # Verify input shape dimensions. To verify that output_stride 16 is possible and not too small 4px
        height, width, _ = input_shape
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError("Height and width of input_shape must be divisible by 64.")
            
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self.eAspp_rate = [3, 6, 12]
        self.res_units = [3, 4, 6, 3]
        self.res_filters = [256, 512, 1024, 2048]
        self.res_strides = [1, 2, 2, 1]
        self.res_dilations = [1, 1, 1, 1]
     
    def _setup(self, data):   
        ### RGB Model 
        self.rgb_b0_out  = self._start_block(data, name="rgb_conv1")
        #block1
        self.rgb_b1_out = self._resnet_unit(self.rgb_b0_out , self.res_filters[0], self.res_strides[0], self.res_dilations[0], "rgb_conv2_block1", identity_connection=False)
        for i in range(2, self.res_units[0]+1):
            self.rgb_b1_out = self._resnet_unit(self.rgb_b1_out, self.res_filters[0], 1, 1, f"rgb_conv2_block{i}")

        #block2
        self.rgb_b2_out = self._resnet_unit(self.rgb_b1_out, self.res_filters[1], self.res_strides[1], self.res_dilations[1], "rgb_conv3_block1", identity_connection=False)
        for i in range(2, self.res_units[1]+1):
            self.rgb_b2_out = self._resnet_unit(self.rgb_b2_out, self.res_filters[1], 1, 1, f"rgb_conv3_block{i}")
       
        #block3
        self.rgb_b3_out = self._resnet_unit(self.rgb_b2_out, self.res_filters[2], self.res_strides[2], self.res_dilations[2], "rgb_conv4_block1", identity_connection=False)
        for i in range(2, self.res_units[2]+1):
            self.rgb_b3_out = self._resnet_unit(self.rgb_b3_out, self.res_filters[2], 1, 1, f"rgb_conv4_block{i}")

        #block4
        self.rgb_b4_out = self._resnet_unit(self.rgb_b3_out, self.res_filters[3], self.res_strides[3], self.res_dilations[3], "rgb_conv5_block1", identity_connection=False)
        for i in range(2, self.res_units[3]+1):
            self.rgb_b4_out = self._resnet_unit(self.rgb_b4_out, self.res_filters[3], 1, 1, f"rgb_conv5_block{i}")
        
        ##skip
        self.rgb_skip1 = self._conv_batchN_relu(self.rgb_b1_out, 1, 1, 24, name="rgb_skip1", relu=False)
        self.rgb_skip2 = self._conv_batchN_relu(self.rgb_b2_out, 1, 1, 24, name="rgb_skip2", relu=False)

        ##eAspp
        self.rgb_eAspp_out = self._eASPP(self.rgb_b4_out, name="rgb_eASPP")

        ### Upsample/Decoder
        self.rgb_deconv_up1 = layers.Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding="same")(self.rgb_eAspp_out)
        self.rgb_deconv_up1 = layers.BatchNormalization()(self.rgb_deconv_up1)

        self.rgb_up1 = self._conv_batchN_relu(tf.concat((self.rgb_deconv_up1, self.rgb_skip2), -1), 3, 1, 256, name="rgb_up1a") 
        self.rgb_up1 = self._conv_batchN_relu(self.rgb_up1, 3, 1, 256, name="rgb_up1b")
            
        self.rgb_deconv_up2 = layers.Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding="same")(self.rgb_up1)
        self.rgb_deconv_up2 = layers.BatchNormalization()(self.rgb_deconv_up2)

        self.rgb_up2 = self._conv_batchN_relu(tf.concat((self.rgb_deconv_up2, self.rgb_skip1), 3), 3, 1, 256, name="rgb_up2a") 
        self.rgb_up2 = self._conv_batchN_relu(self.rgb_up2, 3, 1, 256, name="rgb_up2b")
        self.rgb_up2 = self._conv_batchN_relu(self.rgb_up2, 1, 1, self.num_classes, name="rgb_up1c")

        self.rgb_deconv_up3 = layers.Conv2DTranspose(self.num_classes, kernel_size=8, strides=(4, 4), padding="same")(self.rgb_up2)
        self.rgb_deconv_up3 = layers.BatchNormalization()(self.rgb_deconv_up3)      

        self.rgb_softmax = layers.Softmax(axis=-1, name="rgb_softmax")(self.rgb_deconv_up3)
        out = self.rgb_softmax
        return out
    
    def _start_block(self, x, name):
        outputs = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', name=f'{name}_conv', use_bias=False)(x)
        outputs = layers.BatchNormalization(name=f'{name}_bn')(outputs)
        outputs = layers.Activation("relu", name=f'{name}_relu')(outputs)
        return outputs

    def _resnet_unit(self, x, filters, stride, dilation_factor, name, identity_connection=True):
        assert filters % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name=f'{name}_0_conv')(x)
            o_b1 = layers.BatchNormalization(name=f'{name}_0_bn')(o_b1)
        else:
            o_b1 = x
        # branch2
        o_b2a = layers.Conv2D(filters // 4, kernel_size=1, strides=1, padding='same', name=f'{name}_1_conv')(x)
        o_b2a = layers.BatchNormalization(name=f'{name}_1_bn')(o_b2a)
        o_b2a = layers.Activation("relu", name=f'{name}_1_relu')(o_b2a)

        o_b2b = layers.Conv2D(filters // 4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name=f'{name}_2_conv')(o_b2a)
        o_b2b = layers.BatchNormalization(name=f'{name}_2_bn')(o_b2b)
        o_b2b = layers.Activation("relu", name=f'{name}_2_relu')(o_b2b)

        o_b2c = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name=f'{name}_3_conv')(o_b2b)
        o_b2c = layers.BatchNormalization(name=f'{name}_3_bn')(o_b2c)

        # add
        outputs = layers.Add(name=f'{name}_add')([o_b1, o_b2c])
        # relu
        outputs = layers.Activation("relu", name=f'{name}_out')(outputs)
        return outputs
    
    def _conv_batchN_relu(self, x, kernel_size, stride, filters, name=None, relu=True):
        out = layers.Conv2D(filters, kernel_size, strides=stride, padding="same", name=name, kernel_initializer="he_normal")(x)
        out = layers.BatchNormalization(name=f"{name}_bn")(out)
        if relu:
            out = layers.ReLU(name=f"{name}_relu")(out)
        return out

    def _aconv_batchN_relu(self, x, kernel_size, dilation_rate, filters, name=None, relu=True):
        out = layers.Conv2D(filters, kernel_size, strides=1, dilation_rate=dilation_rate, padding="same", name=name, kernel_initializer="he_normal")(x)
        out = layers.BatchNormalization(name=f"{name}_bn")(out)
        if relu:
            out = layers.ReLU(name=f"{name}_relu")(out)
        return out

    def _eASPP(self, x, name):
        IA = self._conv_batchN_relu(x, 1, 1, 256, name=f"{name}_A")

        IB = self._conv_batchN_relu(x, 1, 1, 64, name=f"{name}_B1")
        IB = self._aconv_batchN_relu(IB, 3, self.eAspp_rate[0], 64, name=f"{name}_B2")
        IB = self._aconv_batchN_relu(IB, 3, self.eAspp_rate[0], 64, name=f"{name}_B3")
        IB = self._conv_batchN_relu(IB, 1, 1, 256, name=f"{name}_B4")

        IC = self._conv_batchN_relu(x, 1, 1, 64, name=f"{name}_C1")
        IC = self._aconv_batchN_relu(IC, 3, self.eAspp_rate[1], 64, name=f"{name}_C2")
        IC = self._aconv_batchN_relu(IC, 3, self.eAspp_rate[1], 64, name=f"{name}_C3")
        IC = self._conv_batchN_relu(IC, 1, 1, 256, name=f"{name}_C4")

        ID = self._conv_batchN_relu(x, 1, 1, 64, name=f"{name}_D1")
        ID = self._aconv_batchN_relu(ID, 3, self.eAspp_rate[2], 64, name=f"{name}_D2")
        ID = self._aconv_batchN_relu(ID, 3, self.eAspp_rate[2], 64, name=f"{name}_D3")
        ID = self._conv_batchN_relu(ID, 1, 1, 256, name=f"{name}_D4")

        IE = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_E1")(x)
        IE = self._conv_batchN_relu(IE, 1, 1, 256, name=f"{name}_E2")
        IE = layers.UpSampling2D(size=x.shape[1] // IE.shape[1], interpolation="bilinear", name=f"{name}_E3")(IE)
        concat = layers.Concatenate(name=f"{name}_add", axis=-1)([IA, IB, IC, ID, IE])

        eAspp_out = self._conv_batchN_relu(concat, 1, 1, 256, name=f"{name}_out", relu=False)
        return eAspp_out

    def _load_pretrained(self, model):
        def remove_prefix(string):
            if string.startswith("rgb_"):
                return string[len("rgb_"):]
            elif string.startswith("depth_"):
                return string[len("depth_"):]
            else:
                return string

        weight_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(self.input_shape[0], self.input_shape[1], 3))

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

    def __call__(self):
        self.inputs_4d = layers.Input(shape=self.input_shape)
        self.outputs = self._setup(self.inputs_4d)
        model = tf.keras.Model(inputs=self.inputs_4d, outputs=self.outputs, name="AdapNet++")
        model = self._load_pretrained(model)
        return model
