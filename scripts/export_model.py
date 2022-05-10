#Import Mask RCNN packages
import mrcnn.model as modellib
import importlib
import tensorflow as tf
import keras.backend as K
import os
from lib import freeze_model, export_saved_model, get_model_size

#Settings
TRANSFORMS = ["remove_nodes(op=Identity)", 
                 "merge_duplicate_nodes",
                 "strip_unused_nodes",
                 "fold_constants(ignore_errors=true)",
                 "fold_batch_norms",
                 # "quantize_nodes", 
                 # "quantize_weights"
                 ]
VERSION_NUMBER = 2

#Get config
# config_path=sys.argv[2]
config_path='./conf/OrganoidBasicConfig20211215.py'
spec = importlib.util.spec_from_file_location('PredictConfig', config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.PredictConfig()

model = modellib.MaskRCNN(mode='inference', config=config, model_dir='./models')
                          #model_dir=config.MODEL_DIR)
model.load_weights('./models/mask_rcnn_organoids_0500.h5', by_name=True)
export_dir = './models/OrganoidBasic20211215/'
if not os.path.exists(export_dir):
    os.mkdir(export_dir)
        
with K.get_session() as master_session:
    graph_def = freeze_model(model.keras_model, master_session)#, transforms = TRANSFORMS)

    with tf.Session(graph = tf.Graph()) as export_session:
        tf.import_graph_def(graph_def, name = "")
        export_saved_model(export_dir, export_session, VERSION_NUMBER)

# Print the size of the tf-serving model
print("*" * 80)
get_model_size(export_dir, VERSION_NUMBER)
print("*" * 80)
print("COMPLETED")