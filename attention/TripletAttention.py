import tensorflow as tf
import math
# All code here are build for Channel Last Achitecture BHWC
######################################################################################################
#                                                                                                    #
# ---------------------------------- Triplet Attention Module  --------------------------------------#
#                                                                                                    #
######################################################################################################
# https://github.com/landskape-ai/triplet-attention
# https://arxiv.org/pdf/2010.03045v2.pdf

class TripletAttention(tf.keras.layers.Layer):
    def __init__(self, no_spatial=False, **kwargs):
        super(TripletAttention, self).__init__(**kwargs)
        self.kernel_size = 7
        self.no_spatial = no_spatial
        self.conv1 = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, strides=1, padding='same')

    def call(self, x):
        # x = BHWC (Channel Last)
        x1 = tf.transpose(x, perm=[0, 3, 2, 1])
        x1_out = self._ZPool(x1) # W pool
        x1_out = self.conv1(x1_out)
        x1_out = tf.keras.activations.sigmoid(x1_out)
        x1_att = x1 * x1_out
        att_branch1 = tf.transpose(x1_att, perm=[0, 3, 2, 1])

        x2 = tf.transpose(x, perm=[0, 1, 3, 2])
        x2_out = self._ZPool(x2) # H pool
        x2_out = self.conv2(x2_out)
        x2_out = tf.keras.activations.sigmoid(x2_out)
        x2_att = x2 * x2_out
        att_branch2 = tf.transpose(x2_att, perm=[0, 1, 3, 2])

        if not self.no_spatial:
            x0_out = self._ZPool(x) # Channel pool
            x0_out = self.conv3(x0_out)
            x0_out = tf.keras.activations.sigmoid(x0_out)
            att_branch0 = x * x0_out
            output = 1 / 3 * (att_branch0 + att_branch1 + att_branch2)
        else:
            output = 1 / 2 * (att_branch1 + att_branch2)
        return output

    def _ZPool(self, inputs):
        # Concatanate Max Pool a Avg Pool accross the W dimension (axis=1)
        return tf.concat([tf.reduce_max(inputs, axis=3, keepdims=True), tf.reduce_mean(inputs, axis=3, keepdims=True)], axis=-1)

    def get_config(self):
        config = {"no_spatial": self.no_spatial}
        base_config = super().get_config()
        return {**base_config, **config}
