import tensorflow as tf
import math
# All code here are build for Channel Last Achitecture BHWC

######################################################################################################
#                                                                                                    #
# ---------------------------------- Coordonate Attention Module ------------------------------------#
#                                                                                                    #
######################################################################################################
# https://arxiv.org/pdf/2103.02907v1.pdf
# https://github.com/houqb/CoordAttention/blob/main/coordatt.py
# @inproceedings{hou2021coordinate,
#   title={Coordinate attention for efficient mobile network design},
#   author={Hou, Qibin and Zhou, Daquan and Feng, Jiashi},
#   booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
#   pages={13713--13722},
#   year={2021}
# }

class CoordAtt(tf.keras.layers.Layer):
    def __init__(self, nb_channels, reduction=32, **kwargs):
        super(CoordAtt, self).__init__(**kwargs)
        self.reduction = reduction
        self.nb_channels = nb_channels
        mip = max(8, self.nb_channels // self.reduction)

        self.relu6 = tf.keras.layers.ReLU(max_value=6)
        self.conv_h = tf.keras.layers.Conv2D(filters=self.nb_channels, kernel_size=1, strides=1, padding='same')
        self.conv_w = tf.keras.layers.Conv2D(filters=self.nb_channels, kernel_size=1, strides=1, padding='same')
        self.conv1 = tf.keras.layers.Conv2D(filters=mip, kernel_size=1, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

    def call(self, x):
        identity = x
        n,h,w,nb_channels = x.shape
        assert h == w, "Width and Height must be the size"

        x_h = tf.math.reduce_mean(x, axis=1, keepdims=True)
        x_w = tf.math.reduce_mean(x, axis=2, keepdims=True)
        x_w = tf.transpose(x_w, perm=[0,2,1,3])
        y = tf.concat([x_h, x_w], axis=2)

        y = self.conv1(y)
        y = self.bn1(y)
        # HSwish
        relu6_output = self.relu6(y+3)/6
        y = y * relu6_output
        x_h, x_w = tf.split(y, [h, w], axis=2)
        a_h = tf.sigmoid(self.conv_h(x_h))
        a_w = tf.sigmoid(self.conv_w(x_w))
        out = identity * a_w * a_h
        return out

    def get_config(self):
        config = {"nb_channels":self.nb_channels, "reduction":self.reduction}
        base_config = super().get_config()
        return {**base_config, **config}
