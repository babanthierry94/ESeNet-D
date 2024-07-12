# https://github.com/zanilzanzan/FuseNet_PyTorch/blob/master/models/fusenet_model.py

# @inproceedings{hazirbas2017fusenet,
#   title={Fusenet: Incorporating depth into semantic segmentation via fusion-based cnn architecture},
#   author={Hazirbas, Caner and Ma, Lingni and Domokos, Csaba and Cremers, Daniel},
#   booktitle={Computer Vision--ACCV 2016: 13th Asian Conference on Computer Vision, Taipei, Taiwan, November 20-24, 2016, Revised Selected Papers, Part I 13},
#   pages={213--228},
#   year={2017},
#   organization={Springer}
# }

import tensorflow as tf
from tensorflow.keras import layers, models

class FuseNet():
    def __init__(self, num_labels, input_shape=(512, 512, 4)):

        self.input4d = tf.keras.layers.Input(shape=input_shape)
        # Load pre-trained VGG-16 weights to two separate variables.
        # They will be used in defining the depth and RGB encoder sequential layers.
        feats_rgb = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
        feats_rgb = add_prefix(feats_rgb, prefix="rgb_")
        feats_depth = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
        feats_depth = add_prefix(feats_depth, prefix="depth_")

        # DEPTH ENCODER
        self.CBR1_D = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, padding='same', name="depth_block1_conv1", kernel_initializer='he_normal'), # VGG16 input has 3channels and Depth image have 1channel
            # feats_depth.get_layer("depth_block1_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block1_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR1_D")
        
        self.CBR2_D = tf.keras.Sequential([
            feats_depth.get_layer("depth_block2_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block2_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR2_D")
        
        self.CBR3_D = tf.keras.Sequential([
            feats_depth.get_layer("depth_block3_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block3_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block3_conv3"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR3_D")
        
        self.dropout3_d = layers.Dropout(0.5)

        self.CBR4_D = tf.keras.Sequential([
            feats_depth.get_layer("depth_block4_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block4_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block4_conv3"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR4_D")
        
        self.dropout4_d = layers.Dropout(0.5)

        self.CBR5_D = tf.keras.Sequential([
            feats_depth.get_layer("depth_block5_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block5_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block5_conv3"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR5_D")

        # RGB ENCODER
        self.CBR1_RGB = tf.keras.Sequential([
            feats_rgb.get_layer("rgb_block1_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block1_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR1_RGB")

        self.CBR2_RGB = tf.keras.Sequential([
            feats_rgb.get_layer("rgb_block2_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block2_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR2_RGB")

        self.CBR3_RGB = tf.keras.Sequential([
            feats_rgb.get_layer("rgb_block3_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block3_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block3_conv3"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR3_RGB")
        self.dropout3 = layers.Dropout(0.5)

        self.CBR4_RGB = tf.keras.Sequential([
            feats_rgb.get_layer("rgb_block4_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block4_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block4_conv3"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR4_RGB")
        self.dropout4 = layers.Dropout(0.5)

        self.CBR5_RGB = tf.keras.Sequential([
            feats_rgb.get_layer("rgb_block5_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block5_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block5_conv3"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR5_RGB")
        self.dropout5 = layers.Dropout(0.5)

        # RGB DECODER
        self.CBR5_Dec = tf.keras.Sequential([
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),
        ], name="CBR5_Dec")

        self.CBR4_Dec = tf.keras.Sequential([
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(256, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),
        ], name="CBR4_Dec")

        self.CBR3_Dec = tf.keras.Sequential([
            layers.Conv2D(256, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(256, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),
        ], name="CBR3_Dec")

        self.CBR2_Dec = tf.keras.Sequential([
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="CBR2_Dec")

        self.CBR1_Dec = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(num_labels, kernel_size=3, padding='same'),
        ], name="CBR1_Dec")

        self.upsampling1 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear', name="dec1")
        self.upsampling2 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear', name="dec2")
        self.upsampling3 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear', name="dec3")
        self.upsampling4 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear', name="dec4")
        self.upsampling5 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear', name="dec5")

        self.rgb_maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="rgb_pool1")
        self.rgb_maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="rgb_pool2")
        self.rgb_maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="rgb_pool3")
        self.rgb_maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="rgb_pool4")

        self.depth_maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="depth_pool1")
        self.depth_maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="depth_pool2")
        self.depth_maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="depth_pool3")
        self.depth_maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="depth_pool4")

        self.maxpool5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=None, padding='same', name="pool5")

    def __call__(self):
        rgb_inputs = self.input4d[:,:,:,:3]
        depth_inputs = tf.expand_dims(self.input4d[:,:,:,3], axis=-1)

        # DEPTH ENCODER
        # Stage 1
        x_1 = self.CBR1_D(depth_inputs)
        x = self.depth_maxpool1(x_1)

        # Stage 2
        x_2 = self.CBR2_D(x)
        x = self.depth_maxpool2(x_2)

        # Stage 3
        x_3 = self.CBR3_D(x)
        x = self.depth_maxpool3(x_3)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_D(x)
        x = self.depth_maxpool4(x_4)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_D(x)

        # RGB ENCODER
        # Stage 1
        y = self.CBR1_RGB(rgb_inputs)
        y = tf.add(y, x_1)
        y = self.rgb_maxpool1(y)

        # Stage 2
        y = self.CBR2_RGB(y)
        y = tf.add(y, x_2)
        y = self.rgb_maxpool2(y)

        # Stage 3
        y = self.CBR3_RGB(y)
        y = tf.add(y, x_3)
        y = self.rgb_maxpool3(y)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB(y)
        y = tf.add(y, x_4)
        y = self.rgb_maxpool4(y)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB(y)
        y = tf.add(y, x_5)
        y = self.maxpool5(y)
        y = self.dropout5(y)

        # DECODER
        # Stage 5 dec
        y = self.upsampling5(y)
        y = self.CBR5_Dec(y)

        # Stage 4 dec
        y = self.upsampling4(y)
        y = self.CBR4_Dec(y)

        # Stage 3 dec
        y = self.upsampling3(y)
        y = self.CBR3_Dec(y)

        # Stage 2 dec
        y = self.upsampling2(y)
        y = self.CBR2_Dec(y)

        # Stage 1 dec
        y = self.upsampling1(y)
        y = self.CBR1_Dec(y)

        model = tf.keras.Model(inputs=self.input4d, outputs=y)
        return model


# https://nrasadi.medium.com/change-model-layer-name-in-tensorflow-keras-58771dd6bf1b
def add_prefix(model, prefix: str, custom_objects=None):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary.
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''
    config = model.get_config()
    old_to_new = {}
    new_to_old = {}

    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]

    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]

    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]

    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())

    return new_model
