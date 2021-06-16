#Init logger
import logging
import logging.config
logging.config.fileConfig('conf/logging.conf')
logger = logging.getLogger(__file__)

from lib import load_train_val_data

#inputs
train_dir= 'data/20210615/train/'
val_dir = 'data/20210615/train/'
image_filter = '_img'
mask_filter = '_mask_organoid'

#training var
epochs = 500

def main():
    ##Get data
    train_images, train_labels, val_images, val_labels = load_train_val_data(train_dir, val_dir, image_filter, mask_filter)

    

if __name__ == "__main__":
    logger.info('Start training...')
    main()
    logger.info('Training completed!')