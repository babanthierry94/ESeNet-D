import tensorflow as tf
import math
# All code here are build for Channel Last Achitecture BHWC
# https://github.com/pprp/awesome-attention-mechanism-in-cv

######################################################################################################
#                                                                                                    #
# ---------------------------------- SimAM Attention Module  ----------------------------------------#
#                                                                                                    #
######################################################################################################
#  https://github.com/cpuimage/SimAM/blob/main/SimAM.py

# @inproceedings{yang2021simam,
#   title={Simam: A simple, parameter-free attention module for convolutional neural networks},
#   author={Yang, Lingxiao and Zhang, Ru-Yuan and Li, Lida and Xie, Xiaohua},
#   booktitle={International conference on machine learning},
#   pages={11863--11874},
#   year={2021},
#   organization={PMLR}
# }
                                                  
class SimAM(tf.keras.layers.Layer):
    def __init__(self, e_lambda=1e-7, **kwargs):
        super(SimAM, self).__init__(**kwargs)
        self.e_lambda = e_lambda
        self.data_format = tf.keras.backend.image_data_format()

    def call(self, inputs, **kwargs):
        input_shape = inputs.shape
        if self.data_format == "channels_first":
            self.height = input_shape[2]
            self.width = input_shape[3]
        else:
            self.height = input_shape[1]
            self.width = input_shape[2]

        # spatial size
        n = self.width * self.height - 1
        # square of (t - u)
        d = tf.math.square(inputs - tf.math.reduce_mean(inputs, axis=(1, 2), keepdims=True))
        # d.sum() / n is channel variance
        v = tf.math.reduce_sum(d, axis=(1, 2), keepdims=True) / n
        # E_inv groups all importance of X
        E_inv = d / (4 *  tf.maximum(v, self.e_lambda) + 0.5)
        return inputs * tf.keras.activations.sigmoid(E_inv)

    def get_config(self):
        config = {"e_lambda": self.e_lambda}
        base_config = super().get_config()
        return {**base_config, **config}
