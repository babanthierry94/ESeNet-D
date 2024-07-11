######################################################################################################
#                                                                                                    #
# ----------------------------------------- FuseNet Model ------------------------------------------#
#                                                                                                    #
######################################################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from typing import Union, Iterable

# https://github.com/zanilzanzan/FuseNet_PyTorch/blob/master/models/fusenet_model.py

class FuseNet(tf.keras.Model):
    def __init__(self, num_labels, input_shape=(512, 512, 4)):
        super(FuseNet, self).__init__()

        self.input4d = tf.keras.layers.Input(shape=input_shape)
        # Load pre-trained VGG-16 weights to two separate variables.
        # They will be used in defining the depth and RGB encoder sequential layers.
        feats_rgb = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
        feats_rgb = add_prefix(feats_rgb, prefix="rgb_")
        feats_depth = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
        feats_depth = add_prefix(feats_depth, prefix="depth_")

        # Average the first layer of feats_rgb variable, the input-layer weights of VGG-16,
        # over the channel dimension, as depth encoder will be accepting one-dimensional
        # inputs instead of three.
        # avg = tf.reduce_mean(feats_rgb.get_layer("block1_conv1").get_weights()[0], axis=-1)
        # avg = tf.expand_dims(avg, axis=-1)

        bn_moment = 0.1

        # DEPTH ENCODER
        self.conv11d = layers.Conv2D(64, kernel_size=3, padding='same', name="depth_block1_conv1")
        # self.conv11d.build((None, None, None, 1))
        # self.conv11d.set_weights([avg.numpy()])

        self.CBR1_D = tf.keras.Sequential([
            # feats_depth.get_layer("depth_block1_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block1_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.CBR2_D = tf.keras.Sequential([
            feats_depth.get_layer("depth_block2_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_depth.get_layer("depth_block2_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
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
        ])
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
        ])
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
        ])

        # RGB ENCODER
        self.CBR1_RGB = tf.keras.Sequential([
            feats_rgb.get_layer("rgb_block1_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block1_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        self.CBR2_RGB = tf.keras.Sequential([
            feats_rgb.get_layer("rgb_block2_conv1"),
            layers.BatchNormalization(),
            layers.ReLU(),
            feats_rgb.get_layer("rgb_block2_conv2"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

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
        ])
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
        ])
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
        ])
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
        ])

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
        ])

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
        ])

        self.CBR2_Dec = tf.keras.Sequential([
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

        self.CBR1_Dec = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(num_labels, kernel_size=3, padding='same'),
        ])

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


        print('[INFO] FuseNet model has been created')

    def __call__(self):
        rgb_inputs = self.input4d[:,:,:,:3]
        depth_inputs = tf.expand_dims(self.input4d[:,:,:,3], axis=-1)

        # DEPTH ENCODER
        # Stage 1
        x = self.conv11d(depth_inputs)
        x_1 = self.CBR1_D(x)
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

# # Usage example
# model = FuseNet(num_labels=10)()
# model.summary()

######################################################################################################
#                                                                                                    #
# ----------------------------------------- MaxUnpooling2D ------------------------------------------#
#                                                                                                    #
######################################################################################################

# https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/layers/max_unpooling_2d.py
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MaxUnpooling2D operation."""


def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.

    A copy of tensorflow.python.keras.util.

    Args:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.

    Returns:
      A tuple of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != n:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(n)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple

def _calculate_output_shape(input_shape, pool_size, strides, padding):
    """Calculates the shape of the unpooled output."""
    if padding == "VALID":
        output_shape = (
            input_shape[0],
            (input_shape[1] - 1) * strides[0] + pool_size[0],
            (input_shape[2] - 1) * strides[1] + pool_size[1],
            input_shape[3],
        )
    elif padding == "SAME":
        output_shape = (
            input_shape[0],
            input_shape[1] * strides[0],
            input_shape[2] * strides[1],
            input_shape[3],
        )
    else:
        raise ValueError('Padding must be a string from: "SAME", "VALID"')
    return output_shape


def _max_unpooling_2d(updates, mask, pool_size=(2, 2), strides=(2, 2), padding="SAME"):
    """Unpool the outputs of a maximum pooling operation."""
    pool_size_attr = " ".join(["i: %d" % v for v in pool_size])
    strides_attr = " ".join(["i: %d" % v for v in strides])
    experimental_implements = [
        'name: "addons:MaxUnpooling2D"',
        'attr { key: "pool_size" value { list {%s} } }' % pool_size_attr,
        'attr { key: "strides" value { list {%s} } }' % strides_attr,
        'attr { key: "padding" value { s: "%s" } }' % padding,
    ]
    experimental_implements = " ".join(experimental_implements)

    @tf.function(experimental_implements=experimental_implements)
    def func(updates, mask):
        mask = tf.cast(mask, "int32")
        input_shape = tf.shape(updates, out_type="int32")
        input_shape = [updates.shape[i] or input_shape[i] for i in range(4)]
        output_shape = _calculate_output_shape(input_shape, pool_size, strides, padding)

        # Calculates indices for batch, height, width and feature maps.
        one_like_mask = tf.ones_like(mask, dtype="int32")
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype="int32"), shape=batch_shape
        )
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype="int32")
        f = one_like_mask * feature_range

        # Transposes indices & reshape update values to one dimension.
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

    return func(updates, mask)


class MaxUnpooling2D(tf.keras.layers.Layer):
    """Unpool the outputs of a maximum pooling operation.

    This function currently does not support outputs of MaxPoolingWithArgMax in
    following cases:
    - include_batch_in_index equals true.
    - input_shape is not divisible by strides if padding is "SAME".
    - (input_shape - pool_size) is not divisible by strides if padding is "VALID".
    - The max pooling operation results in duplicate values in updates and mask.

    Args:
      updates: The pooling result from max pooling.
      mask: the argmax result corresponds to above max values.
      pool_size: The filter that max pooling was performed with. Default: (2, 2).
      strides: The strides that max pooling was performed with. Default: (2, 2).
      padding: The padding that max pooling was performed with. Default: "SAME".
    Input shape:
      4D tensor with shape: `(batch_size, height, width, channel)`.
    Output shape:
      4D tensor with the same shape as the input of max pooling operation.
    """

    def __init__(
        self,
        pool_size: Union[int, Iterable[int]] = (2, 2),
        strides: Union[int, Iterable[int]] = (2, 2),
        padding: str = "SAME",
        **kwargs,
    ):
        super(MaxUnpooling2D, self).__init__(**kwargs)

        if padding != "SAME" and padding != "VALID":
            raise ValueError('Padding must be a string from: "SAME", "VALID"')

        self.pool_size = normalize_tuple(pool_size, 2, "pool_size")
        self.strides = normalize_tuple(strides, 2, "strides")
        self.padding = padding

    def call(self, updates, mask):
        return _max_unpooling_2d(
            updates,
            mask,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[1]
        return _calculate_output_shape(
            input_shape, self.pool_size, self.strides, self.padding
        )

    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config["pool_size"] = self.pool_size
        config["strides"] = self.strides
        config["padding"] = self.padding
        return config


######################################################################################################
#                                                                                                    #
# ------------------------------------ Add Prefix to Layers -----------------------------------------#
#                                                                                                    #
######################################################################################################
    
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
