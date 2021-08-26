import sys
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
#from skimage.io import imread
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,\
    Activation, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D
import os
import cv2
from glob import glob
# not needed in Kaggle, but required in Jupyter
import random
from ImageDataGenerator import DataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard, LambdaCallback

def main():
    data_dir = r'C:\Users\aadisammeta\Documents\MURA\app\MURA-v1.1'
    base_data_dir = r"C:\Users\aadisammeta\Documents\MURA\app"
    list_files = os.listdir(data_dir)
    print(list_files)
    train_image_paths_csv = os.path.join(data_dir, 'train_image_paths.csv')
    train_images_paths = pd.read_csv(train_image_paths_csv, dtype=str, header=None)
    train_images_paths.columns = ['image_path']

    train_labels = train_images_paths['image_path'].map(lambda x: 1 if 'positive' in x else 0)
    train_images_paths['category'] = train_images_paths['image_path'].apply(lambda x: x.split('/')[2])
    train_images_paths['patientId'] = train_images_paths['image_path'].apply(
        lambda x: x.split('/')[3].replace('patient', ''))
    train_images_paths.head()
    _train_image_path_list = train_images_paths['image_path'].values.tolist()
    train_image_path_list = [os.path.join(base_data_dir, _fn) for _fn in _train_image_path_list]
    train_labels_list = train_labels.values.tolist()

    valid_image_paths_csv = os.path.join(data_dir, "valid_image_paths.csv")
    valid_data_paths = pd.read_csv(valid_image_paths_csv, dtype=str, header=None)
    valid_data_paths.columns = ['image_path']
    _valid_image_path_list = valid_data_paths['image_path'].values.tolist()
    valid_image_path_list = [os.path.join(base_data_dir, _fn) for _fn in _valid_image_path_list]

    val_label = valid_data_paths['image_path'].map(lambda x: 1 if 'positive' in x else 0)
    valid_data_paths['category'] = valid_data_paths['image_path'].apply(lambda x: x.split('/')[2])
    valid_data_paths['patientId'] = valid_data_paths['image_path'].apply(
        lambda x: x.split('/')[3].replace('patient', ''))
    valid_labels_list = val_label.values.tolist()

    # shuffle list
    random.seed(42)
    temp = list(zip(train_image_path_list, train_labels_list))
    random.shuffle(temp)
    train_image_path_list_shuffled, train_labels_list_shuffled = zip(*temp)

    temp = list(zip(valid_image_path_list, valid_labels_list))
    random.shuffle(temp)
    valid_image_path_list_shuffled, valid_labels_list_shuffled = zip(*temp)

    num_train_images = int(len(train_labels_list_shuffled) * 0.8)
    print(num_train_images)

    params = {'dim': (299, 299),
              'batch_size': 5,
              'n_classes': 2,
              'n_channels': 3,
              'shuffle': True}

    # Generators
    training_generator = DataGenerator(train_image_path_list_shuffled[0:num_train_images],
                                       train_labels_list_shuffled[0:num_train_images], **params)
    validation_generator = DataGenerator(valid_image_path_list_shuffled[num_train_images:-1],
                                         valid_labels_list_shuffled[num_train_images:-1], **params)

    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    # for layer in base_model.layers:
    #    layer.trainable = False

    from tf_keras_vis.activation_maximization import ActivationMaximization
    from tf_keras_vis.activation_maximization.callbacks import Progress
    from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D
    from tf_keras_vis.activation_maximization.regularizers import TotalVariation2D, Norm
    from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore

    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.input)
    print(model.output)
    model.compile(Adam(learning_rate=.0001), loss='binary_crossentropy', metrics=['accuracy'])
    base_model_dir = './model'

    layer_name = 'predictions'
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    output_class = [2]

    losses = [
        (ActivationMaximization(layer_dict[layer_name], output_class), 2),
        (LPNorm(model.input), 10),
        (TotalVariation(model.input), 10)
    ]
    opt = Optimizer(model.input, losses)
    opt.minimize(max_iter=100, verbose=True, image_modifiers=[Jitter()], callbacks=[GifGenerator('opt_progress')])

    if os.path.exists(base_model_dir) == 0:
        os.makedirs(base_model_dir)

    logdir = os.path.join(base_model_dir, 'logs')
    if os.path.exists(logdir) == 0:
        os.makedirs(logdir)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(logdir, "cp.ckpt"),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    tbCallback = TensorBoard(log_dir=logdir)

    logfileid = open(os.path.join(base_model_dir, 'logfile.txt'), 'w')
    sys.stdout = logfileid

    model.fit(training_generator,
              steps_per_epoch=len(train_labels_list) // params['batch_size'],
              validation_data=validation_generator, callbacks=[model_checkpoint_callback],
              validation_steps=(len(train_labels_list_shuffled) - num_train_images) // params['batch_size'],
              epochs=100, use_multiprocessing=False,
              verbose=2)

    weights_file = os.path.join(base_model_dir, 'model.h5')
    model.save(weights_file)

    sys.stdout.close()

if __name__ == '__main__':
    main()