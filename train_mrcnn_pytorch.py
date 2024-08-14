from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import spawn
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

# Initialize the Process Group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

# Cleanup the Process Group
def cleanup():
    torch.distributed.destroy_process_group()

# Main function for each process
def main(rank, world_size, config_path, job_id):
    # Setup distributed processing
    setup(rank, world_size)

    # Get config
    spec = importlib.util.spec_from_file_location("TrainConfig", config_path)
    modulevar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulevar)
    config = modulevar.TrainConfig()

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
    model = MaskRCNN(backbone, num_classes=config.NUM_CLASSES).float()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Freeze all layers except the heads
    for param in model.module.backbone.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_head = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Use DistributedSampler to handle the data distribution
    data_train = OrganoidDataset_torch(data_train, 4)
    sampler = DistributedSampler(data_train, num_replicas=world_size, rank=rank)
    data_loader_train = DataLoader(data_train, batch_size=config.BATCH_SIZE, sampler=sampler)




    # Train head only
    train_loop(model=model,
               optimizer=optimizer_head,
               epochs=config.EPOCHS_HEADS,
               data_loader=data_loader_train,
               rank=rank)

    # Unfreeze parameters
    for param in model.module.backbone.parameters():
        param.requires_grad = True
    optimizer_all = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train entire model
    train_loop(model=model,
               optimizer=optimizer_all,
               epochs=config.EPOCHS_ALL_LAYERS,
               data_loader=data_loader_train,
               rank=rank)

    # Cleanup
    cleanup()

def train_loop(model, optimizer, epochs, data_loader, rank):
    for epoch in tqdm(range(epochs)):
        data_loader.sampler.set_epoch(epoch)  # Necessary for shuffling in DistributedSampler
        for batch, labels in tqdm(data_loader):
            batch = [img.to(rank) for img in batch]
            labels = [{k: v.to(rank) for k, v in label.items()} for label in labels]
            loss_dict = model(batch, labels)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        if epoch % 10 == 0:
            job_id = sys.argv[1]
            now = datetime.now()  # current date and time
            timestamp = now.strftime("%Y_%m_%dT%H-%M-%S")
            if rank == 0:  # Only save on one process to avoid corruption
                file_path = f"models/{job_id}/Organoids_{timestamp}_epoch_{epoch}.p"
                torch.save(model.state_dict(), file_path)

if __name__ == "__main__":
    job_id = sys.argv[1]
    config_path = sys.argv[2]

    # Number of GPUs
    world_size = torch.cuda.device_count()
    spawn(main, args=(world_size, config_path, job_id), nprocs=world_size, join=True)
