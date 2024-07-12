import tensorflow as tf
import math
# All code here are build for Channel Last Achitecture BHWC

######################################################################################################
#                                                                                                    #
# ---------------------------------- ECA Attention Module  ----------------------------------------#
#                                                                                                    #
######################################################################################################
# https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py

# @inproceedings{wang2020eca,
#   title={ECA-Net: Efficient channel attention for deep convolutional neural networks},
#   author={Wang, Qilong and Wu, Banggu and Zhu, Pengfei and Li, Peihua and Zuo, Wangmeng and Hu, Qinghua},
#   booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
#   pages={11534--11542},
#   year={2020}
# }

class ECA_Attention(tf.keras.layers.Layer):
    def __init__(self, nb_channels, gamma=2, b=1,**kwargs):
        super(ECA_Attention, self).__init__(**kwargs)
        self.gamma = gamma
        self.b = b
        self.nb_channels = nb_channels
        t = int(abs((math.log(self.nb_channels, 2) + self.b) / self.gamma))
        k_size = t if t % 2 else t + 1
        self.avg = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=k_size, padding='same', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        # feature descriptor on the global spatial information
        y = self.avg(inputs)
        # Two different branches of ECA module
        y = tf.squeeze(y, axis=[1, 2])
        y = tf.expand_dims(y, axis=-1)
        y = tf.transpose(y, perm=[0, 2, 1])
        y = self.conv(y)
        y = tf.transpose(y, perm=[0, 2, 1])
        y = tf.expand_dims(y, axis=1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return inputs * y

    def get_config(self):
        config = {"nb_channels":self.nb_channels, "gamma": self.gamma, "b": self.b}
        base_config = super().get_config()
        return {**base_config, **config}
