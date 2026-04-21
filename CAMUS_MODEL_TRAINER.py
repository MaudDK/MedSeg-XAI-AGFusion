import torch
from torchvision import transforms
import pandas as pd
from datasets.CAMUSData import CamusSegmentationDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from diceloss import DiceBCELoss
from trainer import Trainer
from DualEncoder import DualEncoderModel
from AttentionDualEncoder import AttentionDualEncoderModel
from torchvision.models import ResNet34_Weights, ViT_B_16_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from PretrainedModels import Res34UNet, DeepLabV3Res50UNet, Res34UNetNoSkip, DeepLabV3Res50UNetNoSkip
from VisionTransformer import ViTSegmentationModel, CONFIGS
from TimModels import TimSegmentationModel, ViTSwinSegmentationModel, ViTSwinSkipSegmentationModel
from DuckNet import DuckNet
from WeightedAttentionDualEncoder import WeightedAttentionDualEncoderModel
from AttentionDualEncoderSwin import AttentionDualEncoderSwin
from AttentionDualEncoderRes import AttentionDualEncoderRes
import argparse

IMG_SIZE = 384

def begin_training(epochs, patience, model_name, weights = None, load_best = True):
    EPOCHS = epochs
    PATIENCE = patience
    if model_name == "Res34Unet":
        model = Res34UNet(weights=weights, out_channels=1)
        if load_best:
            state_dict = "./checkpoints/20260126/35408fad/Kvasir_Res34SkipUnet/35408fad_epoch_263_metric_0.3408_Kvasir_Res34SkipUnet.pth"
            model.load_state_dict(torch.load(state_dict))
    elif model_name == "Res34UnetNoSkip":
        model = Res34UNetNoSkip(weights=weights, out_channels=1)

    elif model_name == "DeepLabV3Res50UNetNoSkip":
        model = DeepLabV3Res50UNetNoSkip(weights=weights, out_channels=1)

    elif model_name == "DeepLabV3Res50UNet":
        model = DeepLabV3Res50UNet(weights=weights, out_channels=1)
        if load_best:
            state_dict = "./checkpoints/20260126/85621796/Kvasir_DeepLabV3Res50UNet/85621796_epoch_19_metric_0.3321_Kvasir_DeepLabV3Res50UNet.pth"
            model.load_state_dict(torch.load(state_dict))
    
    elif model_name == "ViT_Tiny":
        model = ViTSegmentationModel(num_classes=1, **CONFIGS["ViT_Tiny"])
        if load_best:
            state_dict = "./checkpoints/ViT_Tiny/20260202/Augmentations/f6c6682a_epoch_406_metric_0.4725_ViT_Tiny.pth"
            model.load_state_dict(torch.load(state_dict))

    elif model_name == "ViT_Small":
        model = ViTSegmentationModel(num_classes=1, **CONFIGS["ViT_Small"])
        if load_best:
            state_dict = "./checkpoints/ViT_Small/20260202/Augmentations/f7e34de4_epoch_262_metric_0.5118_ViT_Small.pth"
            model.load_state_dict(torch.load(state_dict))
        
    elif model_name == "ViT_Base":
        model = ViTSegmentationModel(num_classes=1, **CONFIGS["ViT_Base"])
        if load_best:
            state_dict = "./checkpoints/ViT_Base/20260202/Augmentations/f9ce4186_epoch_321_metric_0.5127_ViT_Base.pth"
            model.load_state_dict(torch.load(state_dict))

    elif model_name == "Deit_Base":
        model = ViTSegmentationModel(num_classes=1, pretrained=True, model_name='deit_base_patch16_384', **CONFIGS["ViT_Base"])
    
    elif model_name == "Swin_Base":
        model = ViTSwinSegmentationModel(num_classes=1, pretrained=True)
        
    elif model_name == "Swin_Base_Skip":
        model = ViTSwinSkipSegmentationModel(num_classes=1, pretrained=True)
    
    elif model_name == "ViT_Base_Tim":
        model = TimSegmentationModel(num_classes=1, pretrained=True, model_name='vit_base_patch16_384', img_size=IMG_SIZE)

    elif model_name == "ViT_Small_Tim":
        model = TimSegmentationModel(num_classes=1, pretrained=True, model_name='vit_small_patch16_384', img_size=IMG_SIZE)
    
    elif model_name == "ViT_Tiny_Tim":
        model = TimSegmentationModel(num_classes=1, pretrained=True, model_name='vit_tiny_patch16_384', img_size=IMG_SIZE)
    
    elif model_name == "ViT_Large":
        model = ViTSegmentationModel(num_classes=1, **CONFIGS["ViT_Large"])
    
    elif model_name == "ViT_Huge":
        model = ViTSegmentationModel(num_classes=1, **CONFIGS["ViT_Huge"])
    
    elif model_name == "DuckNet":
        model = DuckNet(in_channels=3, num_classes=1, starting_filters=34)
    elif model_name == "DualEncoder":
        model = DualEncoderModel(out_channels=1, freeze_encoders=False)
    elif model_name == "AttDualEncoder":
        model = AttentionDualEncoderModel(out_channels=1, freeze_encoders=False)
    elif model_name == "WeightedAttDualEncoder":
        model = WeightedAttentionDualEncoderModel(out_channels=1, freeze_encoders=False)
    elif model_name == "AttentionDualEncoderSwin":
        model = AttentionDualEncoderSwin(out_channels=1, freeze_encoders=False)
    elif model_name == "AttentionDualEncoderRes":
        model = AttentionDualEncoderRes(out_channels=1, freeze_encoders=False)
    
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    if load_best:
        best_metric = float(state_dict.split("_metric_")[1].split("_")[0])
        print(f"Loaded best model weights for {model_name} from {state_dict}")
    
    best_metric = 1.0  # Default best metric if not loading

    camus_df = pd.read_csv("./datasets/CAMUS/CAMUS_dataset.csvv")

    kvasir_df = pd.read_csv("./datasets/Kvasir-SEG/Kvasir_dataset.csv")

    train_dataset = KvasirSegmentationDataset(kvasir_df[kvasir_df["split"] == "train"], train=True, img_size=IMG_SIZE)
    val_dataset = KvasirSegmentationDataset(kvasir_df[kvasir_df["split"] == "val"], train=False, img_size=IMG_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)

    ADAM_BETAS = (0.9, 0.999)
    ADAM_WEIGHT_DECAY = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), betas=ADAM_BETAS, lr=1e-4, weight_decay=ADAM_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
    criterion = DiceBCELoss()
    trainer = Trainer(model, train_dataloader, val_dataloader, 
                      epochs=EPOCHS, criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
                      model_name=model_name, 
                      best_metric=best_metric if load_best else 1)
    
    trainer.train(patience=PATIENCE)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True, help="Model name: Res34Unet, DeepLabV3Res50UNet, ViTUNet")
    argparser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    argparser.add_argument("--patience", type=int, default=300, help="Early stopping patience")
    args = argparser.parse_args()
    
    model_name = args.model.strip()
    epochs = args.epochs
    patience = args.patience

    weights = {
        "Res34Unet": ResNet34_Weights.DEFAULT,
        "Res34UnetNoSkip": ResNet34_Weights.DEFAULT,
        "DeepLabV3Res50UNetNoSkip": DeepLabV3_ResNet50_Weights.DEFAULT,
        "DeepLabV3Res50UNet": DeepLabV3_ResNet50_Weights.DEFAULT,
    }.get(model_name, None)

    print(weights)

    begin_training(epochs=epochs, patience=patience, model_name=model_name, weights=weights, load_best=False)
