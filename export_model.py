from keras.models import load_model
new_model = tf.keras.models.load_model('my_model.h5')

from pydeepimagej.yaml import BioImageModelZooConfig

model = load_model('')

dij_config = BioImageModelZooConfig(model, MinimumSize)
