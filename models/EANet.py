# https://github.com/MenghaoGuo/EANet/tree/main

# @article{guo2022beyond,
#   title={Beyond self-attention: External attention using two linear layers for visual tasks},
#   author={Guo, Meng-Hao and Liu, Zheng-Ning and Mu, Tai-Jiang and Hu, Shi-Min},
#   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
#   volume={45},
#   number={5},
#   pages={5436--5447},
#   year={2022},
#   publisher={IEEE}
# }
import tensorflow as tf
from ..attention_modules.ExtAtt import ExternalAttention

class EANet():
    """
    ResNet-101, ResNet-50
    output_stride fixed 16
    """
    def __init__(self, nclass, backbone='resnet101', input_shape=(512, 512, 3), finetune=True):
        if backbone not in ['resnet101', 'resnet50']:
            print("backbone_name ERROR! Please input: resnet101, resnet50")
            raise NotImplementedError
        self.input_shape = input_shape
        if finetune :
            self.pretrained = "imagenet"
        else :
            self.pretrained = None
        self.inputs = tf.keras.layers.Input(shape=self.input_shape)
        self.backbone = backbone

        # Verify input shape dimensions. To verify that output_stride 16 is possible and not too small 4px
        height, width, _ = self.input_shape
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError("Height and width of input_shape must be divisible by 64.")
        else :
             self.ext_head = ExternalAttention(height//16, width//16, 512)

        self.fc0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, kernel_size=3, padding="same", kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.1)
        ])

        self.fc1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.1)
        ])

        self.fc2 = tf.keras.layers.Conv2D(nclass, kernel_size=1, padding="same", kernel_initializer='he_normal')        
        self.resize = tf.keras.layers.Resizing(self.input_shape[0], self.input_shape[1], interpolation="bilinear", name="resize")

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

    def __call__(self):
        # Load backbone
        if self.backbone == 'resnet101':
            backbone_model = tf.keras.applications.ResNet101(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            third_block_layer = backbone_model.get_layer(name="conv4_block23_out").output # 32, 32, 1024
        elif self.backbone == 'resnet50':
            backbone_model = tf.keras.applications.ResNet50(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            third_block_layer = backbone_model.get_layer(name="conv4_block6_out").output # 32, 32, 1024

        #block4
        block4 = self._bottleneck_resblock(third_block_layer, 2048, 1, 2, 'conv5_block1', identity_connection=False)
        for i in range(2, 4):
            block4 = self._bottleneck_resblock(block4, 2048, 1, 1, 'conv5_block%d'%i)
        
        x = self.fc0(block4)
        x = self.ext_head(x)
        x = self.fc1(x)
        x = self.fc2(x)
        outputs = self.resize(x)
        
        # Create the EANet model
        model = tf.keras.Model(inputs=self.inputs, outputs=outputs, name='EANet')
        return model
    
