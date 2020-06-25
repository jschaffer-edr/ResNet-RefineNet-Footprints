from tensorflow import keras
from helpers import get_label_info
from customGenerator import customGenerator
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow.compat.v1 as tf
from tensorflow.python.keras import backend as K
from tensorflow_large_model_support import LMS


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth=True
session = tf.Session(config=config)

tf.disable_v2_behavior()

dataset_basepath = Path("SpaceNet/")
train_images = dataset_basepath / 'train'
train_masks = dataset_basepath / 'train_labels'
val_images = dataset_basepath / 'val'
val_masks = dataset_basepath / 'val_labels'
class_dict = dataset_basepath / 'class_dict.csv'

input_shape = (650, 650, 3)
# dense prediction tasks recommend multiples of 32 +1
random_crop = (224, 224, 3)
#random_crop = (638, 638, 3)
batch_size = 1
epochs = 100
validation_images = 10

class_labels, class_colors, num_classes = get_label_info(
    dataset_basepath / "class_dict.csv")

myValGen = customGenerator(batch_size, val_images, val_masks, num_classes,
                           input_shape, dict(), class_colors, random_crop=random_crop)


img, truth = next(myValGen.generator())


def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = tf.constant(weights)
        if not tf.is_tensor(y_pred):
            y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred, from_logits=True) * K.sum(y_true * Kweights, axis=-1)
    return wcce


model = keras.models.load_model(
    "checkpoints/weights.11-1367.44.hdf5", compile=False)
    #"checkpoints/weights.74-2719.15.hdf5", compile=False)

weights = [1.0, 1.5, 0.5]
#model.compile(optimizer = Adam(lr = 1e-4), loss = CategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
model.compile(optimizer=Adam(
    lr=1e-4), loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])

lms_callback= LMS(swapout_threshold=40, swapin_ahead=3, swapin_groupby=2)
#lms_callback.batch_size = batch_size
#lms_callback.run()

y_pred = model.predict(img, callbacks=[lms_callback])


def display(display_list):
    figure = plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

    # return figure


def labelVisualize(y_pred, mask_colors):
    """
    Convert prediction to color-coded image.
    """
    x = np.argmax(y_pred, axis=-1)
    colour_codes = np.array(mask_colors)
    img = colour_codes[x.astype('uint8')]
    return img


print("pred_shape",  tf.keras.backend.int_shape(y_pred))
print("truth_shape",  tf.keras.backend.int_shape(truth))

pred_img = labelVisualize(y_pred[0], class_colors)
target_labelled = labelVisualize(truth[0], class_colors)


display([img[0], target_labelled, pred_img])
