import tensorflow as tf
import math
# All code here are build for Channel Last Achitecture BHWC


######################################################################################################
#                                                                                                    #
# ---------------------------------- DUAL Attention Module  -----------------------------------------#
#                                                                                                    #
######################################################################################################

# https://github.com/junfu1115/DANet/tree/master

""" Position attention module """
class PAM_Module(tf.keras.layers.Layer):
    def __init__(self, in_dim, name=None):
        super(PAM_Module, self).__init__(name=name)
        self.input_dim = in_dim
        height, width, C = self.input_dim
        # Convolution layers for query, key, and value
        self.query_conv = tf.keras.layers.Conv2D(filters=C, kernel_size=1, kernel_initializer='he_normal')
        self.key_conv = tf.keras.layers.Conv2D(filters=C, kernel_size=1, kernel_initializer='he_normal')
        self.value_conv = tf.keras.layers.Conv2D(filters=C, kernel_size=1, kernel_initializer='he_normal')
        # Learnable parameter for adjustment
        self.beta = tf.Variable(initial_value=tf.zeros(1), trainable=True, name="beta")
        # Reshape layers for efficient matrix operations
        self.reshape_q = tf.keras.layers.Reshape((height * width, C), name="reshape_pam1")
        self.reshape_k = tf.keras.layers.Reshape((height * width, C), name="reshape_pam2")
        self.reshape_v = tf.keras.layers.Reshape((height * width, C), name="reshape_pam3")
        self.reshape_last = tf.keras.layers.Reshape((height, width, C), name="reshape_pam4")

    def call(self, x):
        # Operation for query
        proj_query = self.query_conv(x)
        proj_query = self.reshape_q(proj_query)
        proj_query = tf.transpose(proj_query, perm=[0, 2, 1])
        # Operation for key
        proj_key = self.key_conv(x)
        proj_key = self.reshape_k(proj_key)
        # Energy calculation (dot product between query and key)
        energy = tf.matmul(proj_key, proj_query)
        # Softmax normalization to obtain attention weights
        attention = tf.nn.softmax(energy)
        # Operation for value
        proj_value = self.value_conv(x)
        proj_value = self.reshape_v(proj_value)
        # Output calculation based on attention weights
        out = tf.matmul(attention, proj_value)
        out = self.reshape_last(out)
        # Fine-tuning the output with the gamma parameter
        out = self.beta * out + x
        return out
        
    def get_config(self):
        config = {"in_dim":self.input_dim}
        base_config = super().get_config()
        return {**base_config, **config}


    
""" Channel attention module """
class CAM_Module(tf.keras.layers.Layer):
    def __init__(self, in_dim, name=None):
        super(CAM_Module, self).__init__(name=name)
        self.input_dim = in_dim
        height, width, C = self.input_dim
        # Learnable parameter for adjustment
        self.alpha = tf.Variable(initial_value=tf.zeros(1), trainable=True, name="alpha")
        # Reshape layers for efficient matrix operations
        self.reshape_q = tf.keras.layers.Reshape((height * width, C), name="reshape_cam1")
        self.reshape_k = tf.keras.layers.Reshape((height * width, C), name="reshape_cam2")
        self.reshape_v = tf.keras.layers.Reshape((height * width, C), name="reshape_cam3")
        self.reshape_last = tf.keras.layers.Reshape((height, width, C), name="reshape_cam4")

    def call(self, x):
        # Reshape for efficient matrix operations
        proj_query = self.reshape_q(x)
        proj_key = self.reshape_k(x)
        proj_key = tf.transpose(proj_key, perm=[0, 2, 1])
        # Energy calculation (dot product between query and key)
        energy = tf.matmul(proj_query, proj_key)
        # Modify energy for centered attention
        energy_new = tf.expand_dims(tf.reduce_max(energy, axis=-1), axis=-1) - energy
        # Softmax normalization to obtain attention weights
        attention = tf.nn.softmax(energy_new)
        # Operation for value
        proj_value = self.reshape_v(x)
        # Output calculation based on attention weights
        out = tf.matmul(attention, proj_value)
        out = self.reshape_last(out)
        # Fine-tuning the output with the gamma parameter
        out = self.alpha * out + x
        return out
        
    def get_config(self):
        config = {"in_dim":self.input_dim}
        base_config = super().get_config()
        return {**base_config, **config}
    
