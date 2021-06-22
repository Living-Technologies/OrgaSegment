#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

import matplotlib.pyplot as plt
from lib import load_train_val_names, OrganoidGen, UNet
from keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.distribute import MirroredStrategy
from tensorflow.python.client import device_lib
from tensorflow.config import list_physical_devices

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
img_bit_depth = 8

logger.info(f'Check tensorflow backend: {device_lib.list_local_devices()}')
if len(list_physical_devices("GPU")) == 0:
    logger.error(f'No GPUs available')
    # exit(1)
else:
    logger.info(f'Num GPUs Available: {len(list_physical_devices("GPU"))}')

def main():
    ##Build dataset
    train_image_names, train_label_names, val_image_names, val_label_names = load_train_val_names(data_dir, image_filter=image_filter, mask_filter=mask_filter, val_factor=val_factor)

    #Get data generator
    train_gen = OrganoidGen(batch_size, img_size, img_bit_depth, train_image_names, train_label_names)
    val_gen = OrganoidGen(batch_size, img_size, img_bit_depth, val_image_names, val_label_names)

    ##Build model and use all available GPU's
    strategy = MirroredStrategy()
    logger.info(f'Number of devices: {strategy.num_replicas_in_sync}')
    with strategy.scope():

        model = UNet(input_size = (None, None, 1), 
                     n_filters = 64)
    
    logger.info(model.summary())

    ##Set tensorlogging
    log_dir = "log/fit/" + jobnumber
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)    

    best_weights_file="log/checkpoints/weights.best.hdf5"
    checkpoint = ModelCheckpoint(best_weights_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    callbacks = [tensorboard_callback, checkpoint]

    ##Define steps per epoch
    train_steps = len(train_image_names) // batch_size
    val_steps = len(val_image_names) // batch_size
    
    ##Train model
    model.fit(train_gen, 
              epochs = epochs,
              steps_per_epoch = train_steps,
              validation_data = val_gen,
              validation_steps = val_steps,
              callbacks = callbacks,
              verbose=1)

    logger.info('Saving model...')
    model.save('log/models/' + jobnumber)

if __name__ == "__main__":
    logger.info('Start training...')
    main()
    logger.info('Training completed!')