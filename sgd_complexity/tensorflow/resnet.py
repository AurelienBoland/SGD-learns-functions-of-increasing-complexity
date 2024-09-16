import tensorflow as tf
from tensorflow.keras import layers, models, initializers
import keras_cv

# inpired by https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py and https://github.com/hiyouga/AMP-Regularizer/blob/master/models/preresnet.py

initializers.he_normal()

def basic_block(input, planes, stride=1, num_linear_blocks=0):
    x = layers.BatchNormalization()(input)
    if num_linear_blocks == 0:
        x = layers.Activation('relu')(x)
    
    shorcut = x

    residual = layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', kernel_initializer=initializers.he_normal())(x)
    residual = layers.BatchNormalization()(residual)
    if num_linear_blocks == 0:
        residual = layers.Activation('relu')(residual)
    residual = layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', kernel_initializer=initializers.he_normal())(residual)

    if stride != 1 or planes != shorcut.shape[-1]:
        shorcut = layers.Conv2D(planes, kernel_size=1, strides=stride, padding='same', kernel_initializer=initializers.he_normal())(x)
    
    out = layers.Add()([residual, shorcut])
    return out

def stacked_block(x, planes, num_blocks, stride, num_linear_blocks=0):
    strides = [stride] + [1]*(num_blocks-1)
    for stride in strides:
        x = basic_block(x, planes, stride, num_linear_blocks)
        num_linear_blocks = max(0, num_linear_blocks-1)
    return x

class ResNetV2Augmentation(keras_cv.layers.BaseImageAugmentationLayer):
    def augment_image(self, image, *args, transformation=None, **kwargs):
        image = tf.image.resize_with_pad(image, 40, 40)
        image = tf.image.random_crop(image, (32, 32, 3))
        image = tf.image.random_flip_left_right(image)
        return image
    
        
def resnetV2Cifar(num_blocks, num_linear_blocks=0):
    assert len(num_blocks) <= 4
    input = layers.Input(shape=(32, 32, 3))
    x = ResNetV2Augmentation()(input)
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializers.he_normal())(x)
    x = layers.BatchNormalization()(x)
    if num_linear_blocks == 0:
        x = layers.Activation('relu')(x)
    
    x = stacked_block(x, 16, num_blocks[0], stride=1, num_linear_blocks=num_linear_blocks)
    num_linear_blocks = max(0, num_linear_blocks-num_blocks[0])
    for num_block in num_blocks[1:]:
        x = stacked_block(x, 2*x.shape[-1], num_block, stride=2, num_linear_blocks=num_linear_blocks)
        num_linear_blocks = max(0, num_linear_blocks-num_block)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1)(x)
    x = layers.Activation('sigmoid')(x)
    return models.Model(input, x)

def resnet20():
    return resnetV2Cifar([3,3,3])

def resnet20_with_linear_blocks():
    return resnetV2Cifar([3,3,3], num_linear_blocks=6)


def test(net):
    net.compile()
    total_params = tf.reduce_sum([tf.reduce_prod(x.shape) for x in net.trainable_variables])
    print("Total number of params:", total_params.numpy())
    print("Total layers:", len(net.layers))

if __name__ == "__main__":
    for net_name in ["resnet20", "resnet20_with_linear_blocks"]:
        print(net_name)
        test(globals()[net_name]())
        print()
