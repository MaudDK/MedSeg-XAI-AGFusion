import torch
import torch.nn.functional as F
from TimModels import TimSegmentationModel, ViTSwinSegmentationModel, ViTSwinSkipSegmentationModel
from PretrainedModels import Res34UNet, DeepLabV3Res50UNet, Res34UNetNoSkip, DeepLabV3Res50UNetNoSkip
from VisionTransformer import ViTSegmentationModel, CONFIGS
from DualEncoder import DualEncoderModel
from DuckNet import DuckNet
import numpy as np
from tqdm import tqdm
import pandas as pd
from datasets.KVASIRData import KvasirSegmentationDataset
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

IMG_SIZE = 384

ViTTransformerModels = {
    # "ViT_Tiny_Pretrained": "checkpoints/ViT_Tiny_Tim/20260203/910278fb/910278fb_epoch_48_metric_0.2495_ViT_Tiny_Tim.pth",
    # "ViT_Small_Pretrained": "checkpoints/ViT_Small_Tim/20260203/21b70367/21b70367_epoch_61_metric_0.2350_ViT_Small_Tim.pth",
    # "ViT_Base_Pretrained": "checkpoints/ViT_Base_Tim/20260203/7fe588e8/7fe588e8_epoch_221_metric_0.2440_ViT_Base_Tim.pth",
    # "ViT_Tiny_NoAug": "checkpoints/ViT_Tiny/20260202/Scratch/edd747ef_epoch_79_metric_0.7594_ViT_Tiny.pth",
    # "ViT_Tiny_Aug": "checkpoints/ViT_Tiny/20260202/Augmentations/f6c6682a_epoch_406_metric_0.4725_ViT_Tiny.pth",
    # "ViT_Small_NoAug": "checkpoints/ViT_Small/20260202/Scratch/9ebc8d26_epoch_72_metric_0.7130_ViT_Small.pth",
    # "ViT_Small_Aug": "checkpoints/ViT_Small/20260202/Augmentations/f7e34de4_epoch_262_metric_0.5118_ViT_Small.pth",
    # "ViT_Base_NoAug": "checkpoints/ViT_Base/20260202/Scratch/d9d193a6_epoch_54_metric_0.7601_ViT_Base.pth",
    # "ViT_Base_Aug": "checkpoints/ViT_Base/20260202/Augmentations/f9ce4186_epoch_321_metric_0.5127_ViT_Base.pth",
    # "Swin_Base_Pretrained_Skip": "checkpoints/Swin_Base_Skip/20260203/d699bbf1/d699bbf1_epoch_71_metric_0.2019_Swin_Base_Skip.pth",
    # "Swin_Base_Pretrained_NoSkip": "checkpoints/Swin_Base/20260203/2ef3091a/2ef3091a_epoch_198_metric_0.2125_Swin_Base.pth",
    # "DualEncoder": "checkpoints/DualEncoder/20260205/72f343dc/72f343dc_epoch_56_metric_0.2067_DualEncoder.pth",
    # "AttDualEncoder": "checkpoints/AttDualEncoder/20260206/35f325e3/35f325e3_epoch_62_metric_0.2039_AttDualEncoder.pth",
    # "AttDualEncoderFrozen": "checkpoints/AttDualEncoder/20260206/6d006af5/6d006af5_epoch_91_metric_0.1951_AttDualEncoder.pth",
    # "WeightedAttDualEncoder": "checkpoints/WeightedAttDualEncoder/20260206/61295d5b/61295d5b_epoch_69_metric_0.2044_WeightedAttDualEncoder.pth",
    # "WeightedAttDualEncoderFrozen": "checkpoints/WeightedAttDualEncoder/20260206/d4888481/d4888481_epoch_77_metric_0.1929_WeightedAttDualEncoder.pth",
    "AttentionDualEncoderRes": "checkpoints/AttentionDualEncoderRes/20260206/407ed801/407ed801_epoch_143_metric_0.1941_AttentionDualEncoderRes.pth",
    "AttentionDualEncoderSwin": "checkpoints/AttentionDualEncoderSwin/20260206/c5452230/c5452230_epoch_78_metric_0.1908_AttentionDualEncoderSwin.pth",
    "WeightedDoubleDualEncoderAtt": "checkpoints/AttDualEncoder/20260206/97b8138e/97b8138e_epoch_74_metric_0.1916_AttDualEncoder.pth"
}

CNNModels = {
    "Res34Unet_Aug": "checkpoints/Res34Unet/20260202/Augmentations/53f72439_epoch_280_metric_0.2345_Res34Unet.pth",
    "Res34Unet_Pretrained": "checkpoints/Res34Unet/20260202/Pretrained/4ebb5a3a_epoch_198_metric_0.2078_Res34Unet.pth",
    "Res34Unet_NoAug": "checkpoints/Res34Unet/20260202/Scratch/fefa9154_epoch_58_metric_0.3245_Res34Unet.pth",
    "Res34UnetNoSkip_Aug": "checkpoints/Res34UnetNoSkip/20260202/Augmentations/ad348f6b_epoch_425_metric_0.2214_Res34UnetNoSkip.pth",
    "Res34UnetNoSkip_Pretrained": "checkpoints/Res34UnetNoSkip/20260202/Pretrained/aed6deda_epoch_93_metric_0.2147_Res34UnetNoSkip.pth",
    "DeepLabV3AugNoSkip": "checkpoints/DeepLabV3Res50UNetNoSkip/20260202/Augmentations/54e9fb63_epoch_127_metric_0.2311_DeepLabV3Res50UNetNoSkip.pth",
    "DeepLabV3PretrainedNoSkip": "checkpoints/DeepLabV3Res50UNetNoSkip/20260203/Pretrained/5265394e_epoch_113_metric_0.2315_DeepLabV3Res50UNetNoSkip.pth",
    "DeepLabV3Pretrained": "checkpoints/DeepLabV3Res50UNet/20260202/Pretrained/b18bbbb2_epoch_99_metric_0.2276_DeepLabV3Res50UNet.pth",
    "DeepLabV3Aug": "checkpoints/DeepLabV3Res50UNet/20260202/Augmentations/9f844bbd_epoch_113_metric_0.2195_DeepLabV3Res50UNet.pth",
    "DuckNetAug": "checkpoints/DuckNet/20260204/298fa975/298fa975_epoch_253_metric_0.2219_DuckNet.pth",
    "DuckNetAug34": "checkpoints/DuckNet/20260204/9bc827fb/9bc827fb_epoch_233_metric_0.2292_DuckNet.pth"
}
class MetricsCalculator:
    """Calculate segmentation metrics"""
    @staticmethod
    def binary_cross_entropy_with_logits(logits, target):
        """
        Binary Cross Entropy with logits (numerically stable)
        """
        logits = logits.view(-1)
        target = target.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        return bce.item()
    
    @staticmethod
    def dice_score(pred, target, smooth=1e-6):
        """Calculate Dice Score"""
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()
    
    @staticmethod
    def iou_score(pred, target, smooth=1e-6):
        """Calculate Intersection over Union (IoU / Jaccard Index)"""
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def precision(pred, target, smooth=1e-6):
        """Calculate Precision (Positive Predictive Value)"""
        pred = pred.view(-1)
        target = target.view(-1)
        true_positive = (pred * target).sum()
        predicted_positive = pred.sum()
        precision = (true_positive + smooth) / (predicted_positive + smooth)
        return precision.item()
    
    @staticmethod
    def recall(pred, target, smooth=1e-6):
        """Calculate Recall (Sensitivity / True Positive Rate)"""
        pred = pred.view(-1)
        target = target.view(-1)
        true_positive = (pred * target).sum()
        actual_positive = target.sum()
        recall = (true_positive + smooth) / (actual_positive + smooth)
        return recall.item()
    
    @staticmethod
    def accuracy(pred, target):
        """Calculate Pixel Accuracy"""
        pred = pred.view(-1)
        target = target.view(-1)
        correct = (pred == target).sum()
        total = target.numel()
        accuracy = correct.float() / total
        return accuracy.item()
    @staticmethod
    def f1_score(pred, target, smooth=1e-6):
        precision = MetricsCalculator.precision(pred, target, smooth)
        recall = MetricsCalculator.recall(pred, target, smooth)
        f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)
        return f1   # already a float, OK
    
@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate a single model on all metrics + inference time"""
    if model.__class__.__name__ not in ["AttentionDualEncoderSwin"]:
        return  # Skip evaluation for non-attention models

    model.eval()
    metrics_calc = MetricsCalculator()

    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    all_accuracy = []
    all_f1 = []
    all_bce = []

    inference_times = []  # <-- NEW
    batch = 0

    for images, masks in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        # ----- TIMING START -----
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        outputs = model(images)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        # ----- TIMING END -----

        inference_times.append(end_time - start_time)

        batch_bce = metrics_calc.binary_cross_entropy_with_logits(outputs, masks)
        all_bce.append(batch_bce)

        preds = torch.sigmoid(outputs)
        preds_binary = (preds > 0.5).float()

        if model.__class__.__name__ in ["AttentionDualEncoderSwin"]:
            import os
            import cv2
            os.makedirs(f"attention_weights", exist_ok=True)
            
            attention_weights = model.get_attention_weights()
            
            # Loop through all images in the batch
            for batch_idx in range(images.shape[0]):
                # Get the current image from the batch
                original_image = images[batch_idx].cpu().numpy().transpose(1, 2, 0)
                original_mask = masks[batch_idx].cpu().numpy().transpose(1, 2, 0)
                pred_mask = preds_binary[batch_idx].squeeze().cpu().numpy().astype(np.uint8)

                pred_mask_binary = np.where(pred_mask != 0, 255, 0).astype(np.uint8)
                original_mask_binary = np.where(original_mask.squeeze() != 0, 255, 0).astype(np.uint8)

                masked = np.ma.masked_where(pred_mask_binary == 0, pred_mask_binary)
                o_masked = np.ma.masked_where(original_mask_binary == 0, original_mask_binary)


                # Convert to uint8 for display (assuming images are in [0, 1] range)
                original_image = (original_image * 255).astype(np.uint8)
                
                # Save the attention weights as images for visualization
                weights = []
                for key, weight in attention_weights.items():
                    # Get attention for current batch item
                    if weight.ndim == 4:  # [B, H, W] or [B, ..., H, W]
                        weight_single = weight[batch_idx]
                    else:
                        weight_single = weight

                    weight_single = 1 - weight_single.squeeze().cpu().numpy()
                    
                    # Upsample to 384x384
                    weight_upsampled = cv2.resize(weight_single, (384, 384), interpolation=cv2.INTER_LINEAR)
                    weights.append((key, weight_upsampled))


                # Create figure with 6 subplots
                fig, axes = plt.subplots(1, 6, figsize=(20, 6))
                    
                # 1. Original image + ground truth mask
                axes[0].imshow(original_image)
                axes[0].imshow(o_masked, cmap='Blues_r', alpha=0.5)
                axes[0].set_title('Original', fontsize=12, fontweight='bold')
                axes[0].axis('off')

                #2. Original Image + predicted mask
                axes[1].imshow(original_image)
                axes[1].imshow(masked, cmap='Blues_r', alpha=0.5)
                axes[1].set_title('Predicted', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                # 3. Overlay on original image attention map 1
                axes[2].imshow(original_image)
                axes[2].imshow(weights[0][1], cmap='jet', alpha=0.5)
                axes[2].set_title(f'{weights[0][0]}', fontsize=12, fontweight='bold')
                axes[2].axis('off')

                #4. Overlay on original image attention map 2
                axes[3].imshow(original_image)
                axes[3].imshow(weights[1][1], cmap='jet', alpha=0.5)
                axes[3].set_title(f'{weights[1][0]}', fontsize=12, fontweight='bold')
                axes[3].axis('off')

                #5. Overlay on original image attention map 3
                axes[4].imshow(original_image)
                axes[4].imshow(weights[2][1], cmap='jet', alpha=0.5)
                axes[4].set_title(f'{weights[2][0]}', fontsize=12, fontweight='bold')
                axes[4].axis('off')

                #6. Overlay on original image attention map 4
                axes[5].imshow(original_image)
                axes[5].imshow(weights[3][1], cmap='jet', alpha=0.5)
                axes[5].set_title(f'{weights[3][0]}', fontsize=12, fontweight='bold')
                axes[5].axis('off')

                plt.tight_layout()
                plt.savefig(f"attention_weights/img_{batch}{batch_idx:03d}.png",
                        dpi=150, bbox_inches='tight')
                plt.close()

            print(f"Saved attention visualizations for batch of {images.shape[0]} images")
            batch += 1

        for pred, mask in zip(preds_binary, masks):
            all_dice.append(metrics_calc.dice_score(pred, mask))
            all_iou.append(metrics_calc.iou_score(pred, mask))
            all_precision.append(metrics_calc.precision(pred, mask))
            all_recall.append(metrics_calc.recall(pred, mask))
            all_accuracy.append(metrics_calc.accuracy(pred, mask))
            all_f1.append(metrics_calc.f1_score(pred, mask))

    results = {
        'dice_mean': np.mean(all_dice),
        'dice_std': np.std(all_dice),
        'iou_mean': np.mean(all_iou),
        'iou_std': np.std(all_iou),
        'precision_mean': np.mean(all_precision),
        'precision_std': np.std(all_precision),
        'recall_mean': np.mean(all_recall),
        'recall_std': np.std(all_recall),
        'accuracy_mean': np.mean(all_accuracy),
        'accuracy_std': np.std(all_accuracy),
        'f1_mean': np.mean(all_f1),
        'f1_std': np.std(all_f1),
        'bce_mean': np.mean(all_bce),
        'bce_std': np.std(all_bce),

        # ----- TIME METRICS -----
        'inference_time_mean': np.mean(inference_times),
        'inference_time_std': np.std(inference_times),
    }

    return results

def load_model(model_name, checkpoint_path, device):
    if model_name == "ViT_Tiny_Pretrained":
        model = TimSegmentationModel(num_classes=1, pretrained=False, model_name='vit_tiny_patch16_384', img_size=IMG_SIZE)
    elif model_name == "ViT_Base_Pretrained":
        model = TimSegmentationModel(num_classes=1, pretrained=False, model_name='vit_base_patch16_384', img_size=IMG_SIZE)
    elif model_name == "ViT_Small_Pretrained":
        model = TimSegmentationModel(num_classes=1, pretrained=False, model_name='vit_small_patch16_384', img_size=IMG_SIZE)
    elif model_name == "ViT_Tiny_NoAug" or model_name == "ViT_Tiny_Aug":
        model = ViTSegmentationModel(num_classes=1, **CONFIGS["ViT_Tiny"])
    elif model_name == "ViT_Small_NoAug" or model_name == "ViT_Small_Aug":
        model = ViTSegmentationModel(num_classes=1, **CONFIGS["ViT_Small"])
    elif model_name == "ViT_Base_NoAug" or model_name == "ViT_Base_Aug":
        model = ViTSegmentationModel(num_classes=1, **CONFIGS["ViT_Base"])
    elif model_name == "Swin_Base_Pretrained_Skip":
        model = ViTSwinSkipSegmentationModel(num_classes=1, pretrained=False)
    elif model_name == "Swin_Base_Pretrained_NoSkip":
        model = ViTSwinSegmentationModel(num_classes=1, pretrained=False)
    elif model_name == "Res34Unet_Aug" or model_name == "Res34Unet_Pretrained" or model_name == "Res34Unet_NoAug":
        model = Res34UNet(weights=None, out_channels=1)
    elif model_name == "Res34UnetNoSkip_Aug" or model_name == "Res34UnetNoSkip_Pretrained":
        model = Res34UNetNoSkip(weights=None, out_channels=1)
    elif model_name == "DeepLabV3AugNoSkip" or model_name == "DeepLabV3PretrainedNoSkip":
        model = DeepLabV3Res50UNetNoSkip(weights=None, out_channels=1)
    elif model_name == "DeepLabV3Pretrained" or model_name == "DeepLabV3Aug":
        model = DeepLabV3Res50UNet(weights=None, out_channels=1)
    elif model_name == "DuckNetAug":
        model = DuckNet(in_channels=3, num_classes=1)
    elif model_name == "DuckNetAug34":
        model = DuckNet(in_channels=3, num_classes=1, starting_filters=34)
    elif model_name == "DualEncoder":
        model = DualEncoderModel(out_channels=1, freeze_encoders=False)
    elif model_name == "AttDualEncoder":
        from AttentionDualEncoder import AttentionDualEncoderModel
        model = AttentionDualEncoderModel(out_channels=1, freeze_encoders=True)
    elif model_name == "AttDualEncoderFrozen":
        from AttentionDualEncoder import AttentionDualEncoderModel
        model = AttentionDualEncoderModel(out_channels=1, freeze_encoders=True)
    elif model_name == "WeightedAttDualEncoderFrozen":
        from WeightedAttentionDualEncoder import WeightedAttentionDualEncoderModel
        model = WeightedAttentionDualEncoderModel(out_channels=1, freeze_encoders=True)
    elif model_name == "WeightedAttDualEncoder":
        from WeightedAttentionDualEncoder import WeightedAttentionDualEncoderModel
        model = WeightedAttentionDualEncoderModel(out_channels=1, freeze_encoders=True)
    elif model_name == "AttentionDualEncoderRes":
        from AttentionDualEncoderRes import AttentionDualEncoderRes
        model = AttentionDualEncoderRes(out_channels=1, freeze_encoders=True)
    elif model_name == "AttentionDualEncoderSwin":
        from AttentionDualEncoderSwin import AttentionDualEncoderSwin
        model = AttentionDualEncoderSwin(out_channels=1, freeze_encoders=True)
    elif model_name == "WeightedDoubleDualEncoderAtt":
        from DoubleWeightedDualEncoder import AttentionDualEncoderModel as DoubleWeightedDualEncoderModel
        model = DoubleWeightedDualEncoderModel(out_channels=1, freeze_encoders=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    
    return model

def benchmark_all_models():
    kvasir_df = pd.read_csv("./datasets/Kvasir-SEG/Kvasir_dataset.csv")

    val_dataset = KvasirSegmentationDataset(kvasir_df[kvasir_df["split"] == "val"], train=False, img_size=IMG_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_models = {**ViTTransformerModels, **CNNModels}
    results_list = []

    for model_name, checkpoint_path in all_models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")

        model = load_model(model_name, checkpoint_path, device)
        results = evaluate_model(model, val_dataloader, device)

        if results is None:
            print(f"Skipping {model_name} (not an attention model)")
            continue

        results['model_name'] = model_name
        results['checkpoint'] = checkpoint_path

        # Print results
        print(f"\nResults for {model_name}:")
        print(f"  Dice Score:  {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
        print(f"  IoU:         {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
        print(f"  Precision:   {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"  Recall:      {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"  Accuracy:    {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        print(f"  F1 Score:    {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        print(f"  BCE Loss:    {results['bce_mean']:.4f} ± {results['bce_std']:.4f}")

        results_list.append(results)
        del model
        torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(results_list)
    results_df.sort_values(by='dice_mean', ascending=False, inplace=True)
    results_df.to_csv("benchmark_results.csv", index="model_name")
    print("\nBenchmarking complete. Results saved to 'benchmark_results.csv'.")

if __name__ == "__main__":
    benchmark_all_models()
    