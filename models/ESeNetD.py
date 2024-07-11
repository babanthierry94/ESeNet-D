import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D

class LearnedInterpolation(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, upsampling_factor=2, **kwargs):
        super(LearnedInterpolation, self).__init__(**kwargs)
        self.up_factor = upsampling_factor
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.up2 = tf.keras.layers.UpSampling2D(size=(self.up_factor, self.up_factor), interpolation='bilinear')
        
    def call(self, inputs):
        # Step 1: Fixed interpolation (e.g., bilinear)
        upsampled = self.up2(inputs)
        # Step 2: Learned convolution
        output = self.conv(upsampled)
        return output

class SCConv(tf.keras.layers.Layer):
    def __init__(self, planes, stride, padding, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size=pooling_r, strides=pooling_r),
            tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.k3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.k4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x):
        identity = x
        out_k2 = self.k2(x)
        out_k2 = tf.image.resize(out_k2, size=identity.shape[1:3], method='bilinear')
        out = tf.sigmoid(tf.add(identity, out_k2))
        out = tf.multiply(self.k3(x), out)
        out = self.k4(out)
        return out

class SC_Conv_block(tf.keras.layers.Layer):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, filters, stride=1):
        super(SC_Conv_block, self).__init__()
        group_width = int(filters // 2)
        self.conv1_a = tf.keras.layers.Conv2D(group_width, kernel_size=1, use_bias=False)
        self.bn1_a = tf.keras.layers.BatchNormalization()
        self.relu1_a = tf.keras.layers.ReLU()

        self.conv1_b = tf.keras.layers.Conv2D(group_width, kernel_size=1, use_bias=False)
        self.bn1_b = tf.keras.layers.BatchNormalization()
        self.relu1_b = tf.keras.layers.ReLU()

        self.k1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(group_width, kernel_size=3, strides=stride, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        self.scconv = SCConv(group_width, stride=stride, padding='same', pooling_r=self.pooling_r)
        self.conv3 = tf.keras.layers.Conv2D(filters, kernel_size=1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

    def call(self, x):
        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_a = self.relu1_a(out_a)

        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_b = self.relu1_b(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)

        out = self.conv3(tf.concat([out_a, out_b], axis=-1))
        out = self.bn3(out)
        out = self.relu3(out)
        return out

class Fusion_Block(tf.keras.layers.Layer):
    def __init__(self, nb_channels, reduction_ratio=16, **kwargs):
        super(SE_Attention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.nb_channels = nb_channels
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.excitation = tf.keras.Sequential([
            tf.keras.layers.Dense(self.nb_channels // self.reduction_ratio, activation='relu'),
            tf.keras.layers.Dense(self.nb_channels, activation='sigmoid')
        ])

    def call(self, input1, input2=None):
        if input2 != None :
            features = tf.keras.layers.Concatenate(name='add1')([input1, input2])
        else :
            features = input1
          
        # Squeeze: Global Average Pooling
        squeeze_output = self.squeeze(features)
        # Excitation: Two fully connected layers
        excitation_output = self.excitation(squeeze_output)
        excitation_output = tf.expand_dims(tf.expand_dims(excitation_output, axis=1), axis=1)
        # Scale the input feature map
        return features * excitation_output

    def get_config(self):
        config = {"nb_channels":self.nb_channels, "reduction_ratio": self.reduction_ratio}
        base_config = super().get_config()
        return {**base_config, **config}


class ESeNetD(object):
    """
    EfficientNetv2S
    output_stride fixed 32
    """
    def __init__(self, num_classes=21, finetune=True, input_shape=(512,512,4)):

        self.inputs_4d = tf.keras.layers.Input(shape=input_shape)
        # Input image
        self.input_shape = input_shape
        self.num_classes = num_classes
        if finetune :
            self.pretrained = "imagenet"
        else :
            self.pretrained = None

        # Dilations rates for ASPP module
        self.image_rgb = self.inputs_4d[:,:,:,:3]
        self.depth = tf.expand_dims(self.inputs_4d[:,:,:,3], axis=-1)

        self.backbone_model_RGB = tf.keras.applications.EfficientNetV2S(weights=self.pretrained, include_top=False, input_tensor=self.image_rgb)
        for layer in self.backbone_model_RGB.layers:
            layer._name = str("rgb_") + layer._name

        self.image_depth = tf.concat([self.depth, self.depth, self.depth], -1)
        self.backbone_model_Depth = tf.keras.applications.EfficientNetV2S(weights=self.pretrained, include_top=False, input_tensor=self.image_depth)
        for layer in self.backbone_model_Depth.layers:
            layer._name = str("depth_") + layer._name

        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs_4d, outputs=self.outputs, name='ESeNetD')
        return model

    def build_network(self):
        skip1_RGB = self.backbone_model_RGB.get_layer("rgb_block3d_add").output
        skip2_RGB = self.backbone_model_RGB.get_layer("rgb_block4f_add").output
        skip4_RGB = self.backbone_model_RGB.get_layer("rgb_block6o_add").output

        skip1_D = self.backbone_model_Depth.get_layer("depth_block3d_add").output
        skip2_D = self.backbone_model_Depth.get_layer("depth_block4f_add").output
        skip4_D = self.backbone_model_Depth.get_layer("depth_block6o_add").output
        feature_1 = Fusion_Block(128)(name='add1')([skip1_RGB, skip1_D])
        feature_2 = Fusion_Block(256)(name='add2')([skip2_RGB, skip2_D])
        feature_4 = Fusion_Block(512)(name='add4')([skip4_RGB, skip4_D])

        # Decoder module
        low_features = SC_Conv_block(filters=64)(feature_1)
        middle_features = SC_Conv_block(filters=128)(feature_2)
        middle_features =  tf.keras.layers.UpSampling2D(name="Decoder_Upsampling1a", size=(2,2), interpolation="bilinear")(middle_features) #Upsampling x2
        high_features = SC_Conv_block(filters=256)(feature_4)
        high_features = tf.keras.layers.UpSampling2D(name="Decoder_Upsampling1b", size=(4,4), interpolation="bilinear")(high_features) #Upsampling x4

        x = tf.keras.layers.Concatenate()([high_features, middle_features, low_features])

        x = SC_Conv_block(filters=256)(x)
        x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)
        x = SC_Conv_block(ilters=256)(x)

        x = LearnedInterpolation(filters=self.num_classes)(x)
        x = LearnedInterpolation(filters=self.num_classes)(x)
        self.outputs = x
