from datetime import datetime

import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from lib.io import OrganoidDataset
from tqdm import tqdm
from torchrl.record import CSVLogger
import importlib
import sys
import os
from lib.organoid_dataloader_torch import OrganoidDataset_torch

logger = CSVLogger(exp_name="my_exp")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Get config
config_path = sys.argv[2]
spec = importlib.util.spec_from_file_location("TrainConfig", config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.TrainConfig()

#Set log_dir
log_dir = None

def main():
    job_id = sys.argv[1]
    os.mkdir('models/' + job_id)

    # Get data
    data_train = OrganoidDataset()
    data_train.load_data(config.TRAIN_DIR,
                         config.CLASSES,
                         config.IMAGE_FILTER,
                         config.MASK_FILTER,
                         config.COLOR_MODE)
    data_train.prepare()

    data_val = OrganoidDataset()
    data_val.load_data(config.VAL_DIR,
                       config.CLASSES,
                       config.IMAGE_FILTER,
                       config.MASK_FILTER,
                       config.COLOR_MODE)
    data_val.prepare()

    # Prepare model

    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    model = MaskRCNN(backbone, num_classes=config.NUM_CLASSES).float().to(device=device)


    # Freeze all layers except the heads
    for param in model.backbone.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_head = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)



    data_train=OrganoidDataset_torch(data_train,4,device=device)

    # train head only
    train_loop(model=model,
               optimizer=optimizer_head,
               epochs=config.EPOCHS_HEADS,
               data_train=data_train)

    # Unfreeze parameters
    for param in model.backbone.parameters():
        param.requires_grad = True
    optimizer_all = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train entire model
    train_loop(model=model,
               optimizer=optimizer_all,
               epochs=config.EPOCHS_ALL_LAYERS,
               data_train=data_train)


def train_loop(model, optimizer,epochs,data_train):

    for epoch in tqdm(range(epochs)):
        for item in tqdm(range(len(data_train))):
            batch,labels = data_train[item]
            loss_dict = model(batch, labels)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


        if epoch % 10 == 0:
            job_id = sys.argv[1]
            now = datetime.now()  # current date and time
            timestamp = now.strftime("%Y_%m_%dT%H-%M-%S")
            file_path = f"models/{job_id}/Organoids_{timestamp}_epoch_{epoch}.p"
            torch.save(model.state_dict(), file_path)
        data_train.images_counter= 0


if __name__ == "__main__":
    main()