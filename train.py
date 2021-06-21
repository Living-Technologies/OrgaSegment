#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

import matplotlib.pyplot as plt
from lib import load_train_val_names, OrganoidGen
from keras_unet_collection import models
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.distribute import MirroredStrategy
from tensorflow import keras

#inputs
data_dir= 'data/20210615/train/'
image_filter = '_img'
mask_filter = '_masks_organoid'
jobnumber = 'test001'

#training var
val_factor = 0.20
epochs = 50
batch_size = 5
img_size = (1024, 1024)

def main():
    ##Build dataset
    train_image_names, train_label_names, val_image_names, val_label_names = load_train_val_names(data_dir, image_filter=image_filter, mask_filter=mask_filter, val_factor=val_factor)

    #Get data generator
    train_gen = OrganoidGen(batch_size, img_size, train_image_names, train_label_names)
    val_gen = OrganoidGen(batch_size, img_size, val_image_names, val_label_names)

    ##Build model (use unet lib) and use all available GPU's
    strategy = MirroredStrategy()
    logger.info(f'Number of devices: {strategy.num_replicas_in_sync}')
    with strategy.scope():
        model = models.unet_2d((None, None, 1), [64, 128, 256, 512, 1024], n_labels=2,
                               stack_num_down=2, stack_num_up=1,
                               activation='GELU', output_activation='Softmax', 
                               batch_norm=True, pool='max', unpool='nearest', name='unet')
    
    logger.info(model.summary())

    ##Compile model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=1e-2))

    ##Set tensorlogging
    log_dir = "log/fit/" + jobnumber
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)    

    best_weights_file="log/checkpoints/weights.best.hdf5"
    checkpoint = ModelCheckpoint(best_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    callbacks = [tensorboard_callback, checkpoint]

    ##Train model
    model.fit(train_gen, 
              epochs=epochs,
              validation_data=val_gen,
              callbacks=callbacks,
              verbose=1)

if __name__ == "__main__":
    logger.info('Start training...')
    main()
    logger.info('Training completed!')