import numpy as np
from glob import glob
from src.model import *
from src.utils import *
import tensorflow as tf
from src.cityscapes import *
from datetime import datetime
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler


batch_size = 4

train_dataset = get_dataset(
    glob('./data/train_imgs/*'),
    glob('./data/train_labels/*')
)

val_dataset = get_dataset(
    glob('./data/val_imgs/*'),
    glob('./data/val_labels/*')
)

loss = SparseCategoricalCrossentropy(from_logits=True)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = FastSCNN(n_classes=19)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Conv2DTranspose):
            layer.add_loss(lambda: keras.regularizers.l2(1e-4)(layer.kernel))
    lr_scheduler = PolynomialDecay(5e-4, 1000, 1e-4, 0.9)
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=lr_scheduler)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=logdir, write_graph=True, update_freq='batch')
model_checkpoint = ModelCheckpoint(
    mode='min', filepath='esnet_weights.h5',
    monitor='val_loss', save_best_only='True',
    save_weights_only='True', verbose=1
)

model.fit(
    train_dataset, validation_data=val_dataset, epochs=300,
    steps_per_epoch=len(glob('./data/train_imgs/*')) // batch_size,
    validation_steps=len(glob('./data/val_imgs/*')) // batch_size,
    callbacks=[tensorboard, model_checkpoint]
)
