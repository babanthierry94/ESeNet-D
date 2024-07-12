import tensorflow as tf
from tensorflow.keras import layers

# https://github.com/JindongJiang/RedNet/tree/master
# https://arxiv.org/pdf/1806.01054.pdf

# @article{jiang2018rednet,
#   title={Rednet: Residual encoder-decoder network for indoor rgb-d semantic segmentation},
#   author={Jiang, Jindong and Zheng, Lunan and Luo, Fei and Zhang, Zhijun},
#   journal={arXiv preprint arXiv:1806.01054},
#   year={2018}
# }

class RedNet():
    def __init__(self, num_classes=16, input_shape=(512,512,4)):
        self.input_size = input_shape
        self.inputs = tf.keras.Input(shape=self.input_size)

        # Classifier
        self.classifier = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=2, strides=2, padding='same')

    def __call__(self):
        # ResNet50
        blocks = [3, 4, 6, 3]
        resnet_strides = [1, 2, 2, 2]
        resnet_dilations = [1, 1, 1, 1]

        # Encoder
        rgb_input = self.inputs[:,:,:,:3]
        depth_input = tf.expand_dims(self.inputs[:,:,:,3], axis=-1)

        ################### Depth Branch ##############################
        #starter_block
        block0_depth = self._start_block(depth_input, name='depth_conv1')
        depth_maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="depth_pool1_pool")(block0_depth)
        #block1
        block1_depth = self._bottleneck_resblock(depth_maxpool, 256, resnet_strides[0], resnet_dilations[0], 'depth_conv2_block1', identity_connection=False)
        for i in range(2, blocks[0]+1):
            block1_depth = self._bottleneck_resblock(block1_depth, 256, 1, 1, 'depth_conv2_block%d'%i)
        #block2
        block2_depth = self._bottleneck_resblock(block1_depth, 512, resnet_strides[1], resnet_dilations[1], 'depth_conv3_block1', identity_connection=False)
        for i in range(2, blocks[1]+1):
            block2_depth = self._bottleneck_resblock(block2_depth, 512, 1, 1, 'depth_conv3_block%d'%i)
        #block3
        block3_depth = self._bottleneck_resblock(block2_depth, 1024, resnet_strides[2], resnet_dilations[2], 'depth_conv4_block1', identity_connection=False)
        for i in range(2, blocks[2]+1):
            block3_depth = self._bottleneck_resblock(block3_depth, 1024, 1, 1, 'depth_conv4_block%d'%i)
        #block4
        block4_depth = self._bottleneck_resblock(block3_depth, 2048, resnet_strides[3], resnet_dilations[3], 'depth_conv5_block1', identity_connection=False)
        for i in range(2, blocks[3]+1):
            block4_depth = self._bottleneck_resblock(block4_depth, 2048, 1, 1, 'depth_conv5_block%d'%i)


        ################### RGB Branch ##############################
        #starter_block
        block0_rgb = self._start_block(rgb_input, name='rgb_conv1')
        #block1
        block0_fused = tf.keras.layers.Add()([block0_rgb, block0_depth])
        rgb_maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="rgb_pool1_pool")(block0_fused)
        block1_rgb = self._bottleneck_resblock(rgb_maxpool, 256, resnet_strides[0], resnet_dilations[0], 'rgb_conv2_block1', identity_connection=False)
        for i in range(2, blocks[0]+1):
            block1_rgb = self._bottleneck_resblock(block1_rgb, 256, 1, 1, 'rgb_conv2_block%d'%i)

        #block2
        block1_fused = tf.keras.layers.Add()([block1_rgb, block1_depth])
        block2_rgb = self._bottleneck_resblock(block1_fused, 512, resnet_strides[1], resnet_dilations[1], 'rgb_conv3_block1', identity_connection=False)
        for i in range(2, blocks[1]+1):
            block2_rgb = self._bottleneck_resblock(block2_rgb, 512, 1, 1, 'rgb_conv3_block%d'%i)

        #block3
        block2_fused = tf.keras.layers.Add()([block2_rgb, block2_depth])
        block3_rgb = self._bottleneck_resblock(block2_fused, 1024, resnet_strides[2], resnet_dilations[2], 'rgb_conv4_block1', identity_connection=False)
        for i in range(2, blocks[2]+1):
            block3_rgb = self._bottleneck_resblock(block3_rgb, 1024, 1, 1, 'rgb_conv4_block%d'%i)

        #block4
        block3_fused = tf.keras.layers.Add()([block3_rgb, block3_depth])
        block4_rgb = self._bottleneck_resblock(block3_fused, 2048, resnet_strides[3], resnet_dilations[3], 'rgb_conv5_block1', identity_connection=False)
        for i in range(2, blocks[3]+1):
            block4_rgb = self._bottleneck_resblock(block4_rgb, 2048, 1, 1, 'rgb_conv5_block%d'%i)

        block4_fused = tf.keras.layers.Add()([block4_rgb, block4_depth])

        ################### Decoder Branch ##############################
        agent0 =  self._make_agant_layer(block0_fused, 64, "agent0")
        agent1 =  self._make_agant_layer(block1_fused, 64, "agent1")
        agent2 =  self._make_agant_layer(block2_fused, 128, "agent2")
        agent3 =  self._make_agant_layer(block3_fused, 256, "agent3")
        agent4 =  self._make_agant_layer(block4_fused, 512, "agent4")

        x = agent4
        trans1 = self._make_transpose(x, 256, 6)
        x = tf.keras.layers.Add()([trans1, agent3])

        trans2 = self._make_transpose(x, 128, 4)
        x = tf.keras.layers.Add()([trans2, agent2])

        trans3 = self._make_transpose(x, 64, 3)
        x = tf.keras.layers.Add()([trans3, agent1])

        trans4 = self._make_transpose(x, 64, 3)

        x = tf.keras.layers.Add()([trans4, agent0])

        trans5 = self._make_transpose(x, 64, 3, upsample=False)
        out = self.classifier(trans5)

        model = tf.keras.Model(inputs=self.inputs, outputs=out, name='RedNet')
        model = self._load_pretrained(model)
        return model

    def _start_block(self, x, name):
        outputs = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', name='%s_conv'%name, use_bias=False)(x)
        outputs = tf.keras.layers.BatchNormalization(name='%s_bn'%name)(outputs)
        outputs = tf.keras.layers.Activation("relu", name='%s_relu'%name)(outputs)
        return outputs
    
    def _bottleneck_resblock(self, x, filters, stride, dilation_factor, name, identity_connection=True):
        assert filters % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name='%s_0_conv'%name)(x)
            o_b1 = tf.keras.layers.BatchNormalization(name='%s_0_bn'%name)(o_b1)
        else:
            o_b1 = x
        # branch2
        o_b2a = tf.keras.layers.Conv2D(filters / 4, kernel_size=1, strides=1, padding='same', name='%s_1_conv'%name)(x)
        o_b2a = tf.keras.layers.BatchNormalization(name='%s_1_bn'%name)(o_b2a)
        o_b2a = tf.keras.layers.Activation("relu", name='%s_1_relu'%name)(o_b2a)
    
        o_b2b = tf.keras.layers.Conv2D(filters / 4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name='%s_2_conv'%name)(o_b2a)
        o_b2b = tf.keras.layers.BatchNormalization(name='%s_2_bn'%name)(o_b2b)
        o_b2b = tf.keras.layers.Activation("relu", name='%s_2_relu'%name)(o_b2b)
    
        o_b2c = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name='%s_3_conv'%name)(o_b2b)
        o_b2c = tf.keras.layers.BatchNormalization(name='%s_3_bn'%name)(o_b2c)
    
        # add
        outputs = tf.keras.layers.Add(name='%s_add'%name)([o_b1, o_b2c])
        # relu
        outputs = tf.keras.layers.Activation("relu", name='%s_out'%name)(outputs)
        return outputs
    
    def _make_agant_layer(self, x, filters, name):
        layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ], name=name)
        return layers(x)
    
    def _make_transpose(self, x, filters, nb_blocks, upsample=True):
        x1 = None
        if(not upsample):
            x1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
            x1 = tf.keras.layers.BatchNormalization()(x1)
        else:
            x1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
            x1 = tf.keras.layers.BatchNormalization()(x1)
    
        residual = x
        for i in range(1, nb_blocks):
            residual = x
            block = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization()
            ])(x)
            x = tf.keras.layers.Add()([x, residual])
            x = tf.keras.layers.ReLU()(x)
    
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
        if(not upsample):
            x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        else:
            x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
    
        output = tf.keras.layers.Add()([x, x1])
        output = tf.keras.layers.ReLU()(output)
        return output
    
    def _load_pretrained(self, model):
        def remove_prefix(string):
            if string.startswith("rgb_"):
                return string[len("rgb_"):]
            elif string.startswith("depth_"):
                return string[len("depth_"):]
            else:
                return string

        weight_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(self.input_size[0], self.input_size[1], 3))

        for l in model.layers:
            layer_name = l.name
            if layer_name.startswith(("rgb_", "depth_")):
                layer_name_no_prefix = remove_prefix(layer_name)
                try:
                    if layer_name_no_prefix in [layer.name for layer in weight_model.layers]:
                        weight_layer = weight_model.get_layer(layer_name_no_prefix)
                        l.set_weights(weight_layer.get_weights())
                        # print(f"Loaded weights for layer: {layer_name} (mapped to {layer_name_no_prefix})")
                    else:
                        print(f"No matching layer found in pre-trained model for layer: {layer_name}")
                except Exception as e:
                    print(f"Error loading weights for layer {layer_name}: {str(e)}")
            # else:
            #     print(f"Skipping layer: {layer_name} (not prefixed with 'rgb_' or 'depth_')")
        return model
