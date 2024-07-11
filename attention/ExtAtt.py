import tensorflow as tf
import math
# All code here are build for Channel Last Achitecture BHWC

######################################################################################################
#                                                                                                    #
# ---------------------------------- External Attention Module  -------------------------------------#
#                                                                                                    #
######################################################################################################
class ExternalAttention(tf.keras.layers.Layer):
    def __init__(self, height, width, channels):
        super(ExternalAttention, self).__init__()
        # Hyperparameter for the attention mechanism
        self.k = 64
        # First convolutional layer to process the input
        self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size=1)
        # Linear transformation 0 to compute attention weights
        self.linear_0 = tf.keras.layers.Conv1D(filters=self.k, kernel_size=1, use_bias=False)
        # Linear transformation 1 to apply attention weights to input features
        self.linear_1 = tf.keras.layers.Conv1D(filters=channels, kernel_size=1, use_bias=False)
        # Second convolutional layer and batch normalization for the final output
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(channels, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(synchronized=True)
        ])
        # Reshape layers to manipulate the dimensions of the intermediate tensors
        self.reshape1 = tf.keras.layers.Reshape((height * width, channels))
        self.reshape2 = tf.keras.layers.Reshape((height, width, channels))

    def call(self, inputs):
        # First convolutional layer
        x = self.conv1(inputs)
        # Reshape for attention computation
        x = self.reshape1(x)
        # Linear transformation 0 to compute attention weights
        attn = self.linear_0(x)
        # Apply softmax to get normalized attention weights
        attn = tf.keras.activations.softmax(attn, axis=1)
        # Normalize attention weights
        attn = attn / (1e-9 + tf.reduce_sum(attn, axis=1, keepdims=True))
        # Linear transformation 1 to apply attention weights to input features
        x = self.linear_1(attn)
        # Reshape back to the original dimensions
        x = self.reshape2(x)
        # Second convolutional layer and batch normalization
        x = self.conv2(x)
        # Add the attention-modulated features to the original input
        x = x + inputs
        # Apply ReLU activation
        x = tf.keras.activations.relu(x)

        return x
