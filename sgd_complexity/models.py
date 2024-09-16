import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Add
from tensorflow.keras.utils import plot_model

from sgd_complexity.tensorflow.resnet import resnetV2Cifar

def load_model(name,**kwargs):
    if name == "mnist_demo":
        return mnist_demo_model(**kwargs)
    elif name == "mnist_linear":
        return mnist_linear_model(**kwargs)
    elif name == "mnist_demo_sub":
        return mnist_demo_sub_model(**kwargs)
    elif name == "paper_cnn":
        return paper_cnn_model(**kwargs)
    elif name == "paper_resnet":
        return paper_resnet_model(**kwargs)
    else:
        raise ValueError("Model name not recognized.")

def mnist_demo_model(lr=0.001,plot=False, n_classes=2):
    initializer = "glorot_uniform"
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10,activation='relu', kernel_initializer=initializer))
    if n_classes == 2:
        model.add(Dense(1,activation='sigmoid', kernel_initializer=initializer))
    else:
        model.add(Dense(n_classes,activation='softmax', kernel_initializer=initializer))
    sgd = tf.keras.optimizers.SGD(learning_rate=lr)
    if n_classes == 2:
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if plot:
        plot_model(model, to_file= "model.png",show_shapes=True)
    return model



def paper_cnn_model(lr=0.001,plot=False):
    # Glorot uniform is the default initializer for Conv2D and Dense layers, no need to specify
    model = Sequential()
    for i in range(4):
        model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',))
        if i in [0,1]:
            model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    sgd = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=sgd, 
              loss='binary_crossentropy', metrics=['accuracy'])
    if plot:
        plot_model(model, to_file= "model.png",show_shapes=True)
    return model

def mnist_demo_sub_model(lr=0.001,plot=False):
    initializer = "glorot_uniform"
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10,activation='relu', kernel_initializer=initializer))
    model.add(Dense(1,activation='sigmoid', kernel_initializer=initializer))
    sgd = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if plot:
        plot_model(model, to_file= "model.png",show_shapes=True)
    return model

def mnist_linear_model(lr = 0.001, plot=False, n_classes=2):
    initializer = "glorot_uniform"
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    if n_classes == 2:
        model.add(Dense(1,activation='sigmoid', kernel_initializer=initializer))
    else:
        model.add(Dense(n_classes,activation='softmax', kernel_initializer=initializer))
    sgd = tf.keras.optimizers.SGD(learning_rate=lr)
    if n_classes == 2:
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if plot:
        plot_model(model, to_file= "model_linear.png",show_shapes=True)
    return model

def paper_resnet_model(num_blocks,lr=0.01,num_linear_blocks=0):
    model = resnetV2Cifar(num_blocks, num_linear_blocks)
    sgd = tf.keras.optimizers.SGD(
        learning_rate=lr,
        momentum=0.9,
    )
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    return model
