import tensorflow as tf
import math
# All code here are build for Channel Last Achitecture BHWC

######################################################################################################
#                                                                                                    #
# ---------------------------------- CBAM Attention Module  -----------------------------------------#
#                                                                                                    #
######################################################################################################

# https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
# https://medium.com/dive-into-ml-ai/experimenting-with-convolutional-block-attention-module-cbam-6325a4e2a70f
# https://github.com/kobiso/CBAM-keras/tree/796ae9ea31253d87f46ac4908e94ad5d799fbdd5/
# https://github.com/zhangkaifang/CBAM-TensorFlow2.0/blob/1f8ccd70e7fc127c40d5e059634ad7985046eb14/resnet.py#L15
# https://github.com/MailSuesarn/Convolutional-Block-Attention-Module-CBAM/blob/main/CBAM.py
# https://arxiv.org/pdf/1807.06521.pdf
# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L68
"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
As described in https://arxiv.org/abs/1807.06521.
"""

class CBAM_Attention(tf.keras.layers.Layer):
    def __init__(self, nb_channels, kernel_size=7, ratio=16, **kwargs):
        super(CBAM_Attention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.reduction_ratio = ratio
        # number of features for incoming tensor
        self.nb_channels = nb_channels

        ## Layers for Channel attention
        self.MLP = tf.keras.Sequential([
            tf.keras.layers.Dense(self.nb_channels // self.reduction_ratio, use_bias=True, activation='relu'), # Output shape N,1,1,Channel/ratio
            tf.keras.layers.Dense(self.nb_channels, use_bias=True, kernel_initializer='he_normal') # Output shape N,1,1,Channel
        ])
        self.avgPoolLayer = tf.keras.layers.GlobalAveragePooling2D()
        self.reshapeLayer1 = tf.keras.layers.Reshape((1,1,self.nb_channels))
        self.maxPoolLayer = tf.keras.layers.GlobalMaxPooling2D()
        self.reshapeLayer2 = tf.keras.layers.Reshape((1,1,self.nb_channels))
        self.addLayer = tf.keras.layers.Add()
        self.actLayer = tf.keras.layers.Activation('sigmoid')
        ## Layers for Spatial attention
        self.concat1 = tf.keras.layers.Concatenate(axis=3)
        self.conv1 = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, strides=1, padding='same',
                                              activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    def call(self, input_feature):
        cbam_channel_out = self._CBAM_ChannelAttention(input_feature) 
        cbam_spatial_out = self._CBAM_SpatialAttention(cbam_channel_out) 
        return cbam_spatial_out

    def _CBAM_ChannelAttention(self, input_feature):
        # Global Average Pooling and Global Max Pooling in Spatial domain

        # Average pool over a feature map across channels
        avg_pool = self.avgPoolLayer(input_feature)
        # Output avg_pool shape N,1,1,Channel
        avg_pool = self.reshapeLayer1(avg_pool)

        # Max pool over a feature map across channels
        max_pool = self.maxPoolLayer(input_feature)
        # Output max_pool shape N,1,1,Channel
        max_pool = self.reshapeLayer2(max_pool)

        # shared MLP
        # Output avg_pool_att shape N,1,1,Channel
        avg_pool_att = self.MLP(avg_pool)
        # Output max_pool_att shape N,1,1,Channel
        max_pool_att = self.MLP(max_pool)

        cbam_feature = self.addLayer([avg_pool_att, max_pool_att])
        cbam_feature = self.actLayer(cbam_feature)

        return input_feature * cbam_feature

    def _CBAM_SpatialAttention(self, input_feature):
        # Global Average Pooling and Global Max Pooling in Channel domain
        avg_pool = tf.math.reduce_mean(input_feature, axis=3, keepdims=True)
        max_pool = tf.math.reduce_max(input_feature, axis=3, keepdims=True)
        #Output shape N,H,W,2
        concat = self.concat1([avg_pool, max_pool])
        #Output shape N,H,W,1
        cbam_feature = self.conv1(concat)

        return input_feature * cbam_feature

    def get_config(self):
        config = {"nb_channels":self.nb_channels, "kernel_size": self.kernel_size, "ratio": self.reduction_ratio}
        base_config = super().get_config()
        return {**base_config, **config}
