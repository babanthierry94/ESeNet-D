#########################################################################################################
# Datasets link
# CamerFood15
# https://drive.google.com/drive/u/1/folders/1MugfmVehtIjjyqtphs-4u0GksuHy3Vjz
#########################################################################################################

class CreateDataset_rgbd() :
    """
    Arguments :
        dataset_name  : camerfood10, brazillian, uecfoodpix
        dataset_path  : absolute path of the dataset
        img_size    : input image shape (High, Width, Channel)
        batch_size  : the ratio between input image size and the size of backbone last output features
    """

    def __init__(self, dataset_name, dataset_path, img_shape=(512,512), num_classes=16, batch_size=2, type_data="train"):

        if dataset_name not in ['camerfood15', "nyuv2", "sun", "myfood"]:
            print("ERROR! Dataset name should be : 'camerfood15', 'myfood', 'nyuv2', 'sun'")
            raise NotImplementedError

        # The name of dataset in ['camerfood10', 'brazillian', 'uecfoodpix']
        self.dataset_name = dataset_name
        self.type_data = type_data

        # The dataset absolute path
        self.DATASET_PATH = dataset_path
        if self.dataset_name in ("myfood", "camerfood15"):
            self.DATASET_PATH = os.path.join(dataset_path, self.type_data)
            print(self.DATASET_PATH)
            self.images_path = os.path.join(self.DATASET_PATH, "images")
            self.masks_path = os.path.join(self.DATASET_PATH, "masks")
            self.depth_path = os.path.join(self.DATASET_PATH, "depth/depthAnything")
        elif self.dataset_name in ("nyuv2", "sun"):
            self.ids_filepath = os.path.join(self.DATASET_PATH, "my_" + self.type_data + ".txt")
            self.images_path = os.path.join(self.DATASET_PATH, "RGB")
            self.masks_path = os.path.join(self.DATASET_PATH, "Label")
            self.depth_path = os.path.join(self.DATASET_PATH, "Depth")

        # Dataset parameters
        self.BATCH_SIZE = batch_size
        self.IMAGE_SHAPE = img_shape
        self.NB_CLASS = num_classes

    def load_data(self):
        """
        Returns 2 lists for original and masked files respectively
        """
        # Make a list for images and masks absolute path
        images_list = []
        masks_list = []

        if self.dataset_name == "camerfood15":
            images_list = glob(os.path.join(self.images_path, "*.jpg"))
            masks_list = glob(os.path.join(self.masks_path, "*.png"))
            # depth_list = glob(os.path.join(self.depth_path, "*.png"))
        elif self.dataset_name == "myfood":
            images_list = glob(os.path.join(self.images_path, "*.png"))
            masks_list = glob(os.path.join(self.masks_path, "*.png"))
            # depth_list = glob(os.path.join(self.depth_path, "*.png"))
        elif self.dataset_name in ("nyuv2", "sun") :
            with open(self.ids_filepath, 'r') as f:
                # Read the lines
                files_ids = f.readlines()
                # Construct the file paths for each image ID
                images_list = [self.images_path + '/{}.jpg'.format(file_id.strip()) for file_id in files_ids]
                masks_list = [self.masks_path + '/{}.png'.format(file_id.strip()) for file_id in files_ids]

        images_list = sorted(images_list)
        masks_list = sorted(masks_list)
        print("Number of images:", len(images_list))
        print("Number of masks:", len(masks_list))
        return images_list, masks_list

    # initialization data
    def get(self):

        def read_image(path):
            # Load image
            image = Image.open(path).convert('RGB')
            image = image.resize((self.IMAGE_SHAPE[1], self.IMAGE_SHAPE[0]), resample=Image.BILINEAR)
            image = np.asarray(image)
            image = image[..., :3]
            image = image / 255.0
            image = image.astype(np.float32)
            return image

        def read_mask(path):
            # Load mask
            mask = Image.open(path).convert('L')
            # NEAREST to avoid the problem with pixel value changing in mask
            # Pillow use WHC representation and Tensorflow use HWC for image
            mask = mask.resize((self.IMAGE_SHAPE[1], self.IMAGE_SHAPE[0]), resample=Image.NEAREST)
            mask = np.asarray(mask)

            if self.dataset_name == "camerfood15":
                n = 255 // (self.NB_CLASS-1)
                # n = 255 // 15
                mask = mask / n
            
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(np.uint8)
            return mask

        def read_depth(path):
            # print(path)
            depth_map = Image.open(path).convert('L')
            # Good Code
            depth_map = depth_map.resize((self.IMAGE_SHAPE[1], self.IMAGE_SHAPE[0]), resample=Image.BILINEAR)
            depth_map = np.asarray(depth_map)
            depth_map = depth_map / 255.0
            depth_map = depth_map.astype(np.float32)
            return depth_map

        def preprocess(x, y):
            def f(x_path,y_path):
                # Read true mask
                y_path = y_path.decode()
                y = read_mask(y_path)
                # Read image
                x_path = x_path.decode()
                x = read_image(x_path)
                # Read depth
                z = read_depth(os.path.join(self.depth_path, os.path.basename(y_path)))
                # Build final image
                image_4d = np.append(x, np.expand_dims(z, axis=-1), axis=2)
                return image_4d, y

            images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
            images.set_shape([self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 4])
            masks.set_shape([self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 1])
            return images, masks

        def preprocess_aug1(x, y):
            def f(x_path,y_path):
                # Make a new seed.
                s = (random.randint(1, 65536), 2)
                # Read true mask
                y_path = y_path.decode()
                y = read_mask(y_path)
                # Read image
                x_path = x_path.decode()
                x = read_image(x_path)
                # Read depth
                z = read_depth(os.path.join(self.depth_path, os.path.basename(y_path)))
               
                # Apply image transformations
                # Random brightness.
                x = tf.image.stateless_random_brightness(x, max_delta=0.3, seed=s)
                # Random contrast.
                x = tf.image.stateless_random_contrast(x, lower=0.3, upper=1.0, seed=s)
                # Build final image
                image_4d = np.append(x, np.expand_dims(z, axis=-1), axis=2)
                # Random flip left and right.
                image_4d = tf.image.stateless_random_flip_left_right(image_4d, seed=s)
                y = tf.image.stateless_random_flip_left_right(y, seed=s)
                return image_4d, y

            images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
            images.set_shape([self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 4])
            masks.set_shape([self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 1])
            return images, masks
            
        def preprocess_aug2(x, y):
            def f(x_path,y_path):
                # Make a new seed.
                s = (random.randint(1, 65536), 2)
                # Read true mask
                y_path = y_path.decode()
                y = read_mask(y_path)
                # Read image
                x_path = x_path.decode()
                x = read_image(x_path)
                # Read depth
                z = read_depth(os.path.join(self.depth_path, os.path.basename(y_path)))
                # Apply image transformations
                # Build the 4d image
                image_4d = np.append(x, np.expand_dims(z, axis=-1), axis=2)
                # Random flip left and right.
                image_4d = tf.image.stateless_random_flip_up_down(image_4d, seed=s)
                y = tf.image.stateless_random_flip_up_down(y, seed=s)
                # Random crop
                crop = (self.IMAGE_SHAPE[0]// 4, self.IMAGE_SHAPE[1]//4)
                image_4d = tf.image.stateless_random_crop(image_4d, size=(self.IMAGE_SHAPE[0]-crop[0], self.IMAGE_SHAPE[1]-crop[1], 4), seed=s)
                y = tf.image.stateless_random_crop(y, size=(self.IMAGE_SHAPE[0]-crop[0], self.IMAGE_SHAPE[1]-crop[1], 1), seed=s)
            
                image_4d = tf.image.resize(image_4d, size=(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1]), method='bilinear')
                y = tf.image.resize(y, size=(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1]), method='nearest')
                return image_4d, y
            
            images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
            images.set_shape([self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 4])
            masks.set_shape([self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 1])
            return images, masks
        
        def preprocess_aug3(x, y):
            def f(x_path,y_path):
                # Make a new seed.
                s = (random.randint(1, 65536), 2)
                # Read true mask
                y_path = y_path.decode()
                y = read_mask(y_path)
                # Read image
                x_path = x_path.decode()
                x = read_image(x_path)
                # Read depth
                z = read_depth(os.path.join(self.depth_path, os.path.basename(y_path)))
                # Build final image
                image_4d = np.append(x, np.expand_dims(z, axis=-1), axis=2)
                # Apply image transformations
                # Random flip up and down.
                image_4d = tf.image.stateless_random_flip_up_down(image_4d, seed=s)
                y = tf.image.stateless_random_flip_up_down(y, seed=s)
                # Random flip left and right.
                image_4d = tf.image.stateless_random_flip_left_right(image_4d, seed=s)
                y = tf.image.stateless_random_flip_left_right(y, seed=s)
                
                return image_4d, y

            images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
            images.set_shape([self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 4])
            masks.set_shape([self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 1])
            return images, masks

        images, masks = self.load_data()
        dataset = tf.data.Dataset.from_tensor_slices((images,masks))

        if self.type_data == "train":
            dataset0 = dataset
            dataset1 = tf.data.Dataset.from_tensor_slices((images,masks))
            dataset2 = tf.data.Dataset.from_tensor_slices((images,masks))
            dataset3 = tf.data.Dataset.from_tensor_slices((images,masks))

            dataset0 = dataset0.map(preprocess)
            dataset1 = dataset1.map(preprocess_aug1)
            dataset2 = dataset2.map(preprocess_aug2)
            dataset3 = dataset3.map(preprocess_aug3)

            dataset = dataset0.concatenate(dataset1)
            dataset = dataset.concatenate(dataset2)
            dataset = dataset.concatenate(dataset3)
        else:
            dataset = dataset.map(preprocess)

        dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
