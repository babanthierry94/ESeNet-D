
#####################################################################################################
#                                                                                                    #
# ------------------------------------------- VGG19 UNet --------------------------------------------#
#                                                                                                    #
######################################################################################################
# https://idiotdeveloper.com/vgg19-unet-implementation-in-tensorflow/
# https://idiotdeveloper.com/step-by-step-guide-to-resnet50-unet-in-tensorflow/
# https://idiotdeveloper.com/attention-unet-and-its-implementation-in-tensorflow/
# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
    
def conv_block(input, num_filters):
  # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation = "relu",)(input)
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation = "relu",)(x)
    return x

def decoder_block(input, skip_features, num_filters):
    # upsample
    x = tf.keras.layers.Conv2DTranspose(num_filters, 3, strides=2, padding="same")(input)
    # concatenate
    x = tf.keras.layers.Concatenate()([x, skip_features])
    # Conv2D twice with ReLU activation
    x = conv_block(x, num_filters)
    return x

def VGG16_Unet(input_shape, num_classes):
    """ Input """
    inputs = tf.keras.layers.Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512 x 64)
    s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256 x 128)
    s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128 x 256)
    s4 = vgg16.get_layer("block4_conv3").output         ## (64 x 64 x 512)

    """ Bridge """
    pool4 = vgg16.get_layer("block4_pool").output       ## (32 x 32 x 512)
    bottleneck = conv_block(pool4, 1024)                ## (32 x 32 x 1024)

    """ Decoder """
    d1 = decoder_block(bottleneck, s4, 512)             ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = tf.keras.layers.Conv2D(num_classes, kernel_size=1, padding='same', kernel_initializer='he_normal')(d4)
    model = tf.keras.models.Model(inputs, outputs, name="VGG16_U-Net")
    return model
 
