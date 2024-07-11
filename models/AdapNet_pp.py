import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

# https://github.com/DeepSceneSeg/AdapNet-pp

class AdapNet_pp():
    def __init__(self, num_classes=12, input_shape=(512, 512, 4)):
        # super(AdapNet_pp, self).__init__()

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
            self.rgb_b1_out = self._resnet_unit(self.rgb_b1_out, self.res_filters[0], 1, 1, "rgb_conv2_block%d"%i)

        #block2
        self.rgb_b2_out = self._resnet_unit(self.rgb_b1_out, self.res_filters[1], self.res_strides[1], self.res_dilations[1], "rgb_conv3_block1", identity_connection=False)
        for i in range(2, self.res_units[1]+1):
            self.rgb_b2_out = self._resnet_unit(self.rgb_b2_out, self.res_filters[1], 1, 1, "rgb_conv3_block%d"%i)
       
        #block3
        self.rgb_b3_out = self._resnet_unit(self.rgb_b2_out, self.res_filters[2], self.res_strides[2], self.res_dilations[2], "rgb_conv4_block1", identity_connection=False)
        for i in range(2, self.res_units[2]+1):
            self.rgb_b3_out = self._resnet_unit(self.rgb_b3_out, self.res_filters[2], 1, 1, "rgb_conv4_block%d"%i)

        #block4
        self.rgb_b4_out = self._resnet_unit(self.rgb_b3_out, self.res_filters[3], self.res_strides[3], self.res_dilations[3], "rgb_conv5_block1", identity_connection=False)
        for i in range(2, self.res_units[3]+1):
            self.rgb_b4_out = self._resnet_unit(self.rgb_b4_out, self.res_filters[3], 1, 1, "rgb_conv5_block%d"%i)
        
        ##skip
        self.rgb_skip1 = self._conv_batchN_relu(self.rgb_b1_out, 1, 1, 24, name="rgb_skip1", relu=False)
        self.rgb_skip2 = self._conv_batchN_relu(self.rgb_b2_out, 1, 1, 24, name="rgb_skip2", relu=False)

        ##eAspp
        self.rgb_eAspp_out = self._eASPP(self.rgb_b4_out, name="rgb_eASPP")

        ### Upsample/Decoder
        self.rgb_deconv_up1 = tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding="same")(self.rgb_eAspp_out)
        self.rgb_deconv_up1 = tf.keras.layers.BatchNormalization()(self.rgb_deconv_up1)

        self.rgb_up1 = self._conv_batchN_relu(tf.concat((self.rgb_deconv_up1, self.rgb_skip2), -1), 3, 1, 256, name="rgb_up1a") 
        self.rgb_up1 = self._conv_batchN_relu(self.rgb_up1, 3, 1, 256, name="rgb_up1b")
            
        self.rgb_deconv_up2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding="same")(self.rgb_up1)
        self.rgb_deconv_up2 = tf.keras.layers.BatchNormalization()(self.rgb_deconv_up2)

        self.rgb_up2 = self._conv_batchN_relu(tf.concat((self.rgb_deconv_up2, self.rgb_skip1), 3), 3, 1, 256, name="rgb_up2a") 
        self.rgb_up2 = self._conv_batchN_relu(self.rgb_up2, 3, 1, 256, name="rgb_up2b")
        self.rgb_up2 = self._conv_batchN_relu(self.rgb_up2, 1, 1, self.num_classes, name="rgb_up1c")

        self.rgb_deconv_up3 = tf.keras.layers.Conv2DTranspose(self.num_classes, kernel_size=8, strides=(4, 4), padding="same")(self.rgb_up2)
        self.rgb_deconv_up3 = tf.keras.layers.BatchNormalization()(self.rgb_deconv_up3)      

        self.rgb_softmax = self. tf.keras.layers.Softmax(axis=-1, name="rgb_softmax")(self.rgb_deconv_up3)
        out = self.rgb_softmax
        return out
    
    def _start_block(self, x, name):
        outputs = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', name='%s_conv'%name, use_bias=False)(x)
        outputs = tf.keras.layers.BatchNormalization(name='%s_bn'%name)(outputs)
        outputs = tf.keras.layers.Activation("relu", name='%s_relu'%name)(outputs)
        return outputs

    def _resnet_unit(self, x, filters, stride, dilation_factor, name, identity_connection=True):
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
    
    def _conv_batchN_relu(self, x, kernel_size, stride, filters, name=None, relu=True):
        out = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding="same", name=name, kernel_initializer="he_normal")(x)
        out = tf.keras.layers.BatchNormalization(name="%s_bn"%name)(out)
        if relu:
            out = tf.keras.layers.ReLU(name="%s_relu_"%name)(out)
        return out

    def _aconv_batchN_relu(self, x, kernel_size, dilation_rate, filters, name=None, relu=True):
        out = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, dilation_rate=dilation_rate, padding="same", name=name, kernel_initializer="he_normal")(x)
        out = tf.keras.layers.BatchNormalization(name="%s_bn"%name)(out)
        if relu:
            out = tf.keras.layers.ReLU(name="%s_relu"%name)(out)
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
        IE = tf.keras.layers.UpSampling2D(size=x.shape[1] // IE.shape[1], interpolation="bilinear", name="%s_E3"%name)(IE)
        concat = tf.keras.layers.Concatenate(name="%s_add"%name, axis=-1)([IA, IB, IC, ID, IE])

        eAspp_out = self._conv_batchN_relu(concat, 1, 1, 256, name="%s_out"%name, relu=False)
        return eAspp_out

    def _load_pretrained(self, model) :

        def remove_prefix(string):
            if string.startswith("rgb_"):
                return string[len("rgb_"):]
            elif string.startswith("depth_"):
                return string[len("depth_"):]
            else:
                return string

        weight_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(512, 512, 3))

        for l in model.layers:
            layer_name = l.name
            if(layer_name.startswith(("rgb_", "depth_"))):
                layer_name = remove_prefix(layer_name)
                if(layer_name in weight_model.layers):
                    weight_layer = weight_model.get_layer(layer_name)
                    l.set_weights(weight_layer.get_weights())
                    # print(layer_name
        )
        return model

    def __call__(self):
        self.inputs_4d = tf.keras.layers.Input(shape=self.input_shape)
        self.outputs = self._setup(self.inputs_4d)
        model = tf.keras.Model(inputs=self.inputs_4d, outputs=self.outputs, name="AdapNet++")
        model = self._load_pretrained(model)
        return model


# model = AdapNet_pp(num_classes=16)()
# model.summary()
