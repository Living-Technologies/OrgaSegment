#Import Mask RCNN packages
import mrcnn.model as modellib

#Import OrgaSegment functions
from lib import mask_projection, array_to_base64, base64_to_array, load_img
import importlib

#Import other packages
import mlflow
import numpy as np

class OrgaSegment(mlflow.pyfunc.PythonModel):
    """
    A custom model that adds a specified value `n` to all columns of the input DataFrame.
    Example: https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/notebooks/introduction.html

    Attributes:
    -----------
    n : int
        The value to add to input columns.
    """

    def __init__(self, config):
        """
        Constructor method. Initializes the OrgaSegment model.

        Parameters:
        -----------
        config_path : MASK-RCNN python config
            The relative path of the config to use.
        """
        #load model
        self.model = modellib.MaskRCNN(mode='inference', 
                              config=config,
                              model_dir=config.MODEL_DIR)
        self.model.load_weights(config.MODEL, by_name=True)

    def predict(self, model_input, params=None):
        """
        Prediction method for the OrgaSegment model.

        Parameters:
        -----------
        model_input : json string
            The input DataFrame to which `n` should be added.

        params : dict, optional
            Additional prediction parameters. Ignored in this example.

        Returns:
        --------
        list
            List of dicts per class
        """
        
        # decode numpy array image from base64
        img = base64_to_array(model_input)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        else:
            img = np.max(img, axis=2, keepdims=True)

        #Predict organoids
        pred = model.detect([img], verbose=1)
        # p = pred[0]

        # #Process results per class
        # output = []
        # for c in np.unique(p['class_ids']):
        #     #Get mask
        #     unique_class_ids = (p['class_ids'] == c).nonzero()[0]
        #     mask = mask_projection(p['masks'][:,:,unique_class_ids])

        #     #Process predictions
        #     masks = []
        #     for count, l in enumerate(unique_class_ids):
        #         #Get mask information
        #         msk = p['masks'][:,:,l].astype(np.uint8)
        #         size = np.sum(msk)

        #         #Set all information
        #         info = {'id': count,
        #                 'y1': p['rois'][l,0],
        #                 'x1': p['rois'][l,1],
        #                 'y2': p['rois'][l,2],
        #                 'x2': p['rois'][l,3],
        #                 'class': p['class_ids'][l],
        #                 'score': p['scores'][l],
        #                 'size': size}
        #         masks = masks.append(info, ignore_index=True)
        #     result = {'class_id': c, 'mask': array_to_base64(mask), 'masks': masks}
        
        # output.append(result)

        # return output
        return pred
    
##Set MLflow tracking
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("OrgaSegmment")

#Load config
config_path='./conf/OrganoidBasicConfig20211215.py'
spec = importlib.util.spec_from_file_location('PredictConfig', config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.PredictConfig()

# Save the function as a model
model = OrgaSegment(config)
with mlflow.start_run():
    mlflow.pyfunc.log_model("model", python_model=model)
    run_id = mlflow.active_run().info.run_id



#Test
config_path= './conf/OrganoidBasicConfig20211215.py'

spec = importlib.util.spec_from_file_location('PredictConfig', config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.PredictConfig()
config.display()

model = modellib.MaskRCNN(mode='inference', 
                              config=config,
                              model_dir=config.MODEL_DIR)
model.load_weights(config.MODEL, by_name=True)

#Open image
img = np.asarray(load_img('data/example01.jpg', color_mode=config.COLOR_MODE))
print(img)
img = array_to_base64(img)
print(img)
img = base64_to_array(img)
print(img)
if len(img.shape) == 2:
    img = np.expand_dims(img, 2)
else:
    img = np.max(img, axis=2, keepdims=True)
print(img.shape)

#Precict
pred = model.detect([img], verbose=1)

import io
import base64

def array_to_base64(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    encoded = base64.b64encode(buffer.getvalue())
    return encoded.decode('utf-8')

def base64_to_array(base64_string):
    decoded = base64.b64decode(base64_string)
    buffer = io.BytesIO(decoded)
    return np.load(buffer, allow_pickle=True)

import io
import base64
buffer = io.BytesIO()
np.save(buffer, img)
encoded = base64.b64encode(buffer.getvalue())
print(encoded.decode('utf-8'))