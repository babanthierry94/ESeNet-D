import tensorflow as tf


# @inproceedings{chen2018encoder,
#   title={Encoder-decoder with atrous separable convolution for semantic image segmentation},
#   author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
#   booktitle={Proceedings of the European conference on computer vision (ECCV)},
#   pages={801--818},
#   year={2018}
# }

class DeepLabv3p():
    """
    ResNet-101, ResNet-50
    output_stride fixed 16
    """
    def __init__(self, num_classes=21, backbone="resnet101", input_shape=(512,512,3), finetune=True):
        if backbone not in ['resnet101', 'resnet50']:
            print("backbone_name ERROR! Please input: resnet101, resnet50")
            raise NotImplementedError
        # Verify input shape dimensions. To verify that output_stride 16 is possible and not too small 4px
        height, width, _ = input_shape
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError("Height and width of input_shape must be divisible by 64.")
            
        self.inputs = tf.keras.layers.Input(shape=input_shape)
        self.backbone_name = backbone
        self.num_classes = num_classes
        if finetune :
            self.pretrained = "imagenet"
        else :
            self.pretrained = None
        # Dilations rates for ASPP module
        self.aspp_dilations = [1, 6, 12, 18]
        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='DeepLabv3p')
        return model

    def build_network(self):
        low_level_feat, high_level_feat = self.build_encoder()
        self.outputs = self.build_decoder(low_level_feat, high_level_feat)

    def build_encoder(self):
        if self.backbone_name == 'resnet50':
            backbone_model = tf.keras.applications.ResNet50(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            first_layer_name = "conv2_block3_out" 
            last_layer_name = "conv4_block6_out" 

        elif self.backbone_name == 'resnet101':
            backbone_model = tf.keras.applications.ResNet101(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            first_layer_name = "conv2_block3_out" 
            last_layer_name = "conv4_block23_out"
        #block4
        third_block_layer = backbone_model.get_layer(last_layer_name).output
        block4 = self._bottleneck_resblock(third_block_layer, 2048, 1, 2, 'conv5_block1', identity_connection=False)
        for i in range(2, 4):
            block4 = self._bottleneck_resblock(block4, 2048, 1, 1, 'conv5_block%d'%i)
        high_level_feat = block4

        low_level_feat = backbone_model.get_layer(first_layer_name).output
        # high_level_feat = backbone_model.get_layer(last_layer_name).output
        return low_level_feat, high_level_feat

    def build_decoder(self, low_level_feat, high_level_feat):
        # ASPP module
        high_features = self._ASPPv2(high_level_feat, 256, self.aspp_dilations)

        high_features = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal')(high_features)
        high_features = tf.keras.layers.BatchNormalization()(high_features)
        high_features = tf.keras.layers.Activation('relu')(high_features)
        high_features = tf.keras.layers.UpSampling2D(size=(4,4), interpolation="bilinear")(high_features) #Upsampling x4

        low_features = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', kernel_initializer='he_normal')(low_level_feat)
        low_features = tf.keras.layers.BatchNormalization()(low_features)
        low_features = tf.keras.layers.Activation('relu')(low_features)

        x = tf.keras.layers.Concatenate()([high_features, low_features])

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.UpSampling2D(size=self.inputs.shape[1]//x.shape[1], interpolation="bilinear")(x)
        outputs = tf.keras.layers.Conv2D(self.num_classes, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

        return outputs
    
    # ResNet Bottleneck Block
    def _bottleneck_resblock(self, x, filters, stride, dilation_factor, name, identity_connection=True):
        """Defines a ResNet bottleneck block."""
        assert filters % 4 == 0, 'Bottleneck number of output ERROR!'
        
        if not identity_connection:
            o_b1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name=f'{name}_1_conv')(x)
            o_b1 = tf.keras.layers.BatchNormalization(name=f'{name}_1_bn')(o_b1)
        else:
            o_b1 = x

        o_b2a = tf.keras.layers.Conv2D(filters//4, kernel_size=1, strides=1, padding='same', name=f'{name}_2_conv')(x)
        o_b2a = tf.keras.layers.BatchNormalization(name=f'{name}_2_bn')(o_b2a)
        o_b2a = tf.keras.layers.Activation("relu", name=f'{name}_2_relu')(o_b2a)

        o_b2b = tf.keras.layers.Conv2D(filters//4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name=f'{name}_3_conv')(o_b2a)
        o_b2b = tf.keras.layers.BatchNormalization(name=f'{name}_3_bn')(o_b2b)
        o_b2b = tf.keras.layers.Activation("relu", name=f'{name}_3_relu')(o_b2b)

        o_b2c = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name=f'{name}_4_conv')(o_b2b)
        o_b2c = tf.keras.layers.BatchNormalization(name=f'{name}_4_bn')(o_b2c)

        outputs = tf.keras.layers.Add(name=f'{name}_add')([o_b1, o_b2c])
        outputs = tf.keras.layers.Activation("relu", name=f'{name}_out')(outputs)
        return outputs
    
    def _Atrous_SepConv(self, x, conv_type="conv2d", prefix="None", filters=256, kernel_size=3,  stride=1, dilation_rate=1, use_bias=False):
        conv_dict = {
            'conv2d': tf.keras.layers.Conv2D,
            'sepconv2d': tf.keras.layers.SeparableConv2D
        }
        conv = conv_dict[conv_type]
        x = conv(filters, kernel_size, name=prefix, strides=stride, dilation_rate=dilation_rate,
                                padding="same", kernel_initializer='he_normal', use_bias=use_bias)(x)
        x = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn")(x)
        x = tf.keras.layers.Activation('relu', name=f"{prefix}_relu")(x)
        return x

    def _ASPPv2(self, x, nb_filters, d):
        x1 = self._Atrous_SepConv(x, conv_type="conv2d", prefix='aspp/sepconv1', filters=nb_filters, kernel_size=1, dilation_rate=d[0], use_bias=True)
        x2 = self._Atrous_SepConv(x, conv_type="conv2d", prefix='aspp/sepconv2', filters=nb_filters, kernel_size=3, dilation_rate=d[1], use_bias=True)
        x3 = self._Atrous_SepConv(x, conv_type="conv2d", prefix='aspp/sepconv3', filters=nb_filters, kernel_size=3, dilation_rate=d[2], use_bias=True)
        x4 = self._Atrous_SepConv(x, conv_type="conv2d", prefix='aspp/sepconv4', filters=nb_filters, kernel_size=3, dilation_rate=d[3], use_bias=True)

        x5 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='aspp/avg')(x)
        x5 = tf.keras.layers.Conv2D(256, kernel_size=1)(x5)
        x5 = tf.keras.layers.UpSampling2D(size=x.shape[1] // x5.shape[1], interpolation="bilinear", name='asspv2/avg_upsambling')(x5)
        out = tf.keras.layers.Concatenate(name='aspp/add')([x1, x2, x3, x4, x5])
        return out
