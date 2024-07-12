import tensorflow as tf
import math
# All code here are build for Channel Last Achitecture BHWC
######################################################################################################
#                                                                                                    #
# -------------------------------------- SE Attention Module ----------------------------------------#
#                                                                                                    #
######################################################################################################
# https://github.com/hujie-frank/SENet
# https://arxiv.org/pdf/1709.01507v4.pdf

# @inproceedings{hu2018squeeze,
#   title={Squeeze-and-excitation networks},
#   author={Hu, Jie and Shen, Li and Sun, Gang},
#   booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
#   pages={7132--7141},
#   year={2018}
# }

class SE_Attention(tf.keras.layers.Layer):
    def __init__(self, nb_channels, reduction_ratio=16, **kwargs):
        super(SE_Attention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.nb_channels = nb_channels
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.excitation = tf.keras.Sequential([
            tf.keras.layers.Dense(self.nb_channels // self.reduction_ratio, activation='relu'),
            tf.keras.layers.Dense(self.nb_channels, activation='sigmoid')
        ])

    def call(self, inputs):
        # Squeeze: Global Average Pooling
        squeeze_output = self.squeeze(inputs)

        # Excitation: Two fully connected layers
        excitation_output = self.excitation(squeeze_output)
        excitation_output = tf.expand_dims(tf.expand_dims(excitation_output, axis=1), axis=1)

        # Scale the input feature map
        return inputs * excitation_output

    def get_config(self):
        config = {"nb_channels":self.nb_channels, "reduction_ratio": self.reduction_ratio}
        base_config = super().get_config()
        return {**base_config, **config}
