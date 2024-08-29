import importlib
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.resnet import ResNet101_Weights
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm

from lib.io import OrganoidDataset
from lib.organoid_dataloader_torch import OrganoidDataset_torch

# Initialize TensorBoard SummaryWriter
log_dir = f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get config
config_path = sys.argv[2]
spec = importlib.util.spec_from_file_location("TrainConfig", config_path)
modulevar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulevar)
config = modulevar.TrainConfig()


def main():
    job_id = sys.argv[1]
    save_folder = create_unique_folder(job_id)

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
    model = prepare_model()

    # Freeze all layers except the heads
    for param in model.backbone.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_head = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)

    data_train = OrganoidDataset_torch(data_train, config.BATCH_SIZE)
    data_val = OrganoidDataset_torch(data_val, 1)

    total_epoch = 0
    # Train head only
    total_epoch = train_loop(model=model,
                             optimizer=optimizer_head,
                             epochs=config.EPOCHS_HEADS,
                             data_train=data_train,
                             save_folder=save_folder,
                             total_epochs=total_epoch,
                             data_val=data_val)

    # Unfreeze parameters
    for param in model.backbone.parameters():
        param.requires_grad = True
    optimizer_all = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    # Train entire model
    train_loop(model=model,
               optimizer=optimizer_all,
               epochs=config.EPOCHS_ALL_LAYERS,
               data_train=data_train,
               save_folder=save_folder,
               total_epochs=total_epoch,
               data_val=data_val)

    now = datetime.now()  # current date and time
    timestamp = now.strftime("%Y_%m_%d")
    file_path = f"models/{save_folder}/Organoids_{timestamp}_END.p"
    torch.save(model.state_dict(), file_path)

    # Close TensorBoard writer
    writer.close()


def train_loop(model, optimizer, epochs, data_train, save_folder, total_epochs, data_val):
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        loss_classifier = 0
        loss_box_reg = 0
        loss_mask = 0
        loss_objectness = 0
        loss_rpn_box_reg = 0

        for item in range(len(data_train)):
            batch, labels = data_train[item]
            batch = [batch.to(device=device) for batch in batch]
            labels = [{k: v.to(device) for k, v in label.items()} for label in labels]

            loss_dict = model(batch, labels)

            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            loss_classifier = loss_dict['loss_classifier'].item()
            loss_box_reg = loss_dict['loss_box_reg'].item()
            loss_mask = loss_dict['loss_mask'].item()
            loss_objectness = loss_dict['loss_objectness'].item()
            loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()

        # Log metrics to TensorBoard
        avg_loss = epoch_loss / (len(data_train) * data_train.batch_size)
        avg_loss_classifier = loss_classifier / (len(data_train) * data_train.batch_size)
        avg_loss_box_reg = loss_box_reg / (len(data_train) * data_train.batch_size)
        avg_loss_mask = loss_mask / (len(data_train) * data_train.batch_size)
        avg_loss_objectness = loss_objectness / (len(data_train) * data_train.batch_size)
        avg_loss_rpn_box_reg = loss_rpn_box_reg / (len(data_train) * data_train.batch_size)

        writer.add_scalar('Loss/train', avg_loss, total_epochs)
        writer.add_scalar('Loss/classifier', avg_loss_classifier, total_epochs)
        writer.add_scalar('Loss/box_reg', avg_loss_box_reg, total_epochs)
        writer.add_scalar('Loss/mask', avg_loss_mask, total_epochs)
        writer.add_scalar('Loss/objectness', avg_loss_objectness, total_epochs)
        writer.add_scalar('Loss/rpn_box_reg', avg_loss_rpn_box_reg, total_epochs)

        if epoch % 10 == 0:
            val_loss = 0
            try:
                with torch.no_grad():
                    val_indices = np.random.choice(len(data_val), 5, replace=False)
                    for val_index in val_indices:
                        batch, labels = data_val[val_index]
                        batch = [batch.to(device=device) for batch in batch]
                        labels = [{k: v.to(device) for k, v in label.items()} for label in labels]
                        loss_dict = model(batch, labels)
                        losses = sum(loss for loss in loss_dict.values())
                        val_loss += losses.item()

                avg_val_loss = val_loss / len(data_val)
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
            except Exception:
                print(traceback.print_exc())
                print('indices', val_indices)
                print('Failed index', val_index)

            now = datetime.now()  # current date and time
            timestamp = now.strftime("%Y_%m_%d")
            file_path = f"models/{save_folder}/Organoids_{timestamp}_epoch_{total_epochs}.p"
            torch.save(model.state_dict(), file_path)
            # Log model checkpoint to TensorBoard
            writer.add_text('Checkpoint', f"Model saved at {file_path}", epoch)
        data_train.images_counter = 0
        data_val.images_counter = 0
        total_epochs += 1

    return total_epochs


def prepare_model():
    # Initialize the anchor generator
    anchor_sizes = tuple((size,) for size in config.RPN_ANCHOR_SCALES)

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,  # RPN_ANCHOR_SCALES
        aspect_ratios=(tuple(config.RPN_ANCHOR_RATIOS),) * len(config.RPN_ANCHOR_SCALES)
    )

    # Initialize the backbone with FPN
    backbone = resnet_fpn_backbone(
        'resnet101',
        trainable_layers=3,
        returned_layers=[1, 2, 3, 4],
        weights=ResNet101_Weights.DEFAULT
    )

    # Define the box and mask heads for the RoIHeads
    resolution = config.POOL_SIZE  # POOL_SIZE
    representation_size = config.FPN_CLASSIF_FC_LAYERS_SIZE  # Size of the fully connected layers
    mask_resolution = config.MASK_POOL_SIZE  # MASK_POOL_SIZE

    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                      output_size=resolution,
                                      sampling_ratio=2)

    mask_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                       output_size=mask_resolution,
                                       sampling_ratio=2)

    box_head = TwoMLPHead(
        in_channels=backbone.out_channels * resolution ** 2,
        representation_size=representation_size
    )

    # Predictor head for boxes
    box_predictor = FastRCNNPredictor(
        in_channels=representation_size,
        num_classes=config.NUM_CLASSES  # Number of classes including background
    )

    # Predictor head for masks
    mask_predictor = MaskRCNNPredictor(
        in_channels=backbone.out_channels,
        dim_reduced=256,
        num_classes=config.NUM_CLASSES  # Number of classes including background
    )

    # Initialize RoIHeads
    roi_heads = RoIHeads(
        # ROI pooling settings
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,
        mask_predictor=mask_predictor,
        fg_iou_thresh=0.5,  # Foreground IoU threshold
        bg_iou_thresh=0.5,  # Background IoU threshold
        batch_size_per_image=config.TRAIN_ROIS_PER_IMAGE,  # TRAIN_ROIS_PER_IMAGE
        positive_fraction=config.ROI_POSITIVE_RATIO,  # ROI_POSITIVE_RATIO
        bbox_reg_weights=None,
        score_thresh=config.DETECTION_MIN_CONFIDENCE,  # DETECTION_MIN_CONFIDENCE
        nms_thresh=config.DETECTION_NMS_THRESHOLD,  # DETECTION_NMS_THRESHOLD
        detections_per_img=config.DETECTION_MAX_INSTANCES  # DETECTION_MAX_INSTANCES
    )

    # Construct the MaskRCNN model
    model = MaskRCNN(
        backbone=backbone,
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=config.PRE_NMS_LIMIT,  # PRE_NMS_LIMIT
        rpn_pre_nms_top_n_test=config.PRE_NMS_LIMIT,  # PRE_NMS_LIMIT
        rpn_post_nms_top_n_train=config.POST_NMS_ROIS_TRAINING,  # POST_NMS_ROIS_TRAINING
        rpn_post_nms_top_n_test=config.POST_NMS_ROIS_INFERENCE,  # POST_NMS_ROIS_INFERENCE
        rpn_nms_thresh=config.RPN_NMS_THRESHOLD,  # RPN_NMS_THRESHOLD
        rpn_fg_iou_thresh=config.ROI_POSITIVE_RATIO,  # ROI_POSITIVE_RATIO
        rpn_bg_iou_thresh=0.3,  # Background IoU threshold for RPN
        rpn_batch_size_per_image=config.RPN_TRAIN_ANCHORS_PER_IMAGE,  # RPN_TRAIN_ANCHORS_PER_IMAGE
        rpn_positive_fraction=config.ROI_POSITIVE_RATIO,  # ROI_POSITIVE_RATIO
        box_roi_pool=roi_heads.box_roi_pool,
        box_head=roi_heads.box_head,
        box_predictor=roi_heads.box_predictor,
        mask_roi_pool=roi_heads.mask_roi_pool,
        mask_head=roi_heads.mask_head,
        mask_predictor=roi_heads.mask_predictor,
        pretrained=True
    ).to(device=device)


    reset_layers = ['backbone.body.conv1.weight',
                    'roi_heads.box_predictor.cls_score.weight',
                    'roi_heads.box_predictor.cls_score.bias',
                    'roi_heads.box_predictor.bbox_pred.weight',
                    'roi_heads.box_predictor.bbox_pred.bias',
                    'roi_heads.mask_predictor.mask_fcn_logits.weight',
                    'roi_heads.mask_predictor.mask_fcn_logits.bias']

    # Reset specified layers
    for layer_name in reset_layers:
        if 'weight' in layer_name:
            torch.nn.init.kaiming_normal_(model.state_dict()[layer_name], mode='fan_out', nonlinearity='relu')
        elif 'bias' in layer_name:
            torch.nn.init.constant_(model.state_dict()[layer_name], 0)

    return model


def create_unique_folder(base_name):
    # Start with the base name
    folder_name = base_name

    # Initialize a counter to track the number of "_new" additions
    counter = 0

    # Keep modifying the folder name until a non-existing one is found
    while os.path.exists('./models/' + folder_name):
        counter += 1
        folder_name = f"{base_name}{'_new' * counter}"

    # Create the folder
    os.makedirs('./models/' + folder_name)
    print(f"Folder created: {folder_name}")
    return folder_name


if __name__ == "__main__":
    main()
