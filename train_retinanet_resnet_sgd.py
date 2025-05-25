import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors
import os
import argparse
from tqdm import tqdm
import time
import warnings
import multiprocessing
import torch.nn as nn
import random
import numpy as np
import functools

try:
    import torchmetrics
    from torchmetrics.detection import MeanAveragePrecision
    torchmetrics_available = True
except ImportError:
    torchmetrics_available = False


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_IDX = {cls_name: i + 1 for i, cls_name in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES) + 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TargetTransformV2:
    def __init__(self, class_to_idx, img_size):
        self.class_to_idx = class_to_idx
        self.img_size = (img_size, img_size)

    def __call__(self, image, target):
        try:
            objects = target['annotation']['object']
            if not isinstance(objects, list):
                objects = [objects]

            boxes = []
            labels = []
            for obj in objects:
                class_name = obj['name']
                if class_name not in self.class_to_idx:
                    continue

                bbox = obj['bndbox']
                xmin = float(bbox['xmin'])
                ymin = float(bbox['ymin'])
                xmax = float(bbox['xmax'])
                ymax = float(bbox['ymax'])

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[class_name])

            if not boxes:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)

            target_dict = {
                'boxes': tv_tensors.BoundingBoxes(boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=image.shape[-2:]),
                'labels': labels
            }
            return image, target_dict
        except Exception as e:
            return image, {
                'boxes': tv_tensors.BoundingBoxes(torch.empty((0, 4), dtype=torch.float32), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=image.shape[-2:]),
                'labels': torch.empty((0,), dtype=torch.int64)
            }

def get_voc_datasets(data_path, year, image_set, img_size):
    common_transforms = [
        transforms.ToImage(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToDtype(torch.float32, scale=True)
    ]

    if image_set in ['train', 'trainval']:
        current_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            *common_transforms,
            TargetTransformV2(class_to_idx=CLASS_TO_IDX, img_size=img_size)
        ])
    else: # 'val' or 'test'
        current_transforms = transforms.Compose([
            *common_transforms,
            TargetTransformV2(class_to_idx=CLASS_TO_IDX, img_size=img_size)
        ])

    try:
        dataset = VOCDetection(
            root=data_path,
            year=year,
            image_set=image_set,
            download=False,
            transforms=current_transforms
        )
        if len(dataset) == 0:
            pass
    except Exception as e:
        print(f"Error: Could not load VOC dataset for {year} {image_set} from {data_path}: {e}. Returning None.")
        return None
    return dataset

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack([img for img in images if img is not None])
    valid_targets = [tgt for tgt in targets if tgt is not None and isinstance(tgt, dict) and 'boxes' in tgt and 'labels' in tgt]
    return images, valid_targets


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, output_dir='outputs_retinanet', metric_name='map_50', model_name_prefix='retinanet_resnet50'):
        self.patience = patience
        self.min_delta = min_delta
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.model_name_prefix = model_name_prefix
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = os.path.join(self.output_dir, f'best_model_{self.model_name_prefix}.pt')
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, val_metrics, model, epoch, optimizer, lr_scheduler, args, scaler=None):
        if not val_metrics or self.metric_name not in val_metrics:
            score = -float('inf')
        else:
            score = val_metrics[self.metric_name].item() if isinstance(val_metrics[self.metric_name], torch.Tensor) else val_metrics[self.metric_name]

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch, optimizer, lr_scheduler, args, score, scaler)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best {self.metric_name}: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, epoch, optimizer, lr_scheduler, args, score, scaler)
            self.counter = 0

    def save_checkpoint(self, model, epoch, optimizer, lr_scheduler, args, score, scaler):
        print(f"Validation {self.metric_name} improved ({self.best_score:.4f} --> {score:.4f}). Saving model to {self.path}...")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            f'val_{self.metric_name}': score
        }
        if lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        if scaler:
            checkpoint['amp_scaler_state_dict'] = scaler.state_dict()
        try:
            torch.save(checkpoint, self.path)
        except Exception as e:
            print(f"Error saving best model checkpoint: {e}")


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, use_amp, args):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")

    for batch_idx, (images, targets) in enumerate(progress_bar):
        if not images.numel() or not targets:
            continue

        images = images.to(device)
        processed_targets = []
        for t in targets:
            if isinstance(t, dict) and 'boxes' in t and 'labels' in t:
                try:
                    boxes = t['boxes'].to(device)
                    labels = t['labels'].to(device)
                    if not isinstance(boxes, tv_tensors.BoundingBoxes):
                         boxes = tv_tensors.BoundingBoxes(
                             boxes,
                             format=tv_tensors.BoundingBoxFormat.XYXY,
                             canvas_size=images[0].shape[-2:]
                         )
                    processed_targets.append({'boxes': boxes, 'labels': labels})
                except Exception as e:
                    pass
            else:
                pass

        if not processed_targets:
            continue

        optimizer.zero_grad()

        try:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss_dict = model(images, processed_targets)
                if not loss_dict:
                    continue
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                optimizer.zero_grad()
                return {"loss": float('inf'), "error": "non-finite loss"}


            if use_amp:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            current_loss = losses.item()
            total_loss += current_loss
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", cls_loss=f"{loss_dict.get('classification', torch.tensor(0.0)).item():.4f}", reg_loss=f"{loss_dict.get('bbox_regression', torch.tensor(0.0)).item():.4f}")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            else:
                return {"loss": float('inf'), "error": str(e)}
        except Exception as e:
            return {"loss": float('inf'), "error": str(e)}

    progress_bar.close()
    if num_batches > 0:
        avg_epoch_loss = total_loss / num_batches
        print(f"  Training Loss: {avg_epoch_loss:.4f}")
        return {"loss": avg_epoch_loss}
    else:
        print("  Training: No batches were processed.")
        return {"loss": 0}


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, total_epochs, args):
    model.eval()
    val_metric = None
    if torchmetrics_available:
        try:
            val_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds=[0.50])
            val_metric = val_metric.to(device)
        except Exception as e:
            print(f"Failed to initialize MeanAveragePrecision: {e}. mAP calculation will be skipped.")
            val_metric = None
    else:
        print("torchmetrics not available, mAP calculation will be skipped.")

    progress_bar_val = tqdm(data_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Validation]")

    for images, targets in progress_bar_val:
        if not images.numel() or not targets:
            continue

        images = images.to(device)

        formatted_targets = []
        for i, t in enumerate(targets):
            if isinstance(t, dict) and 'boxes' in t and 'labels' in t:
                 target_boxes = t['boxes']
                 if not isinstance(target_boxes, tv_tensors.BoundingBoxes):
                      target_boxes = tv_tensors.BoundingBoxes(
                          target_boxes,
                          format=tv_tensors.BoundingBoxFormat.XYXY,
                          canvas_size=images[0].shape[-2:]
                      )

                 formatted_targets.append({
                     'boxes': target_boxes.to(device),
                     'labels': t['labels'].to(device)
                 })
            else:
                pass
        if not formatted_targets: continue

        try:
            predictions = model(images)
            formatted_preds = []
            for p in predictions:
                formatted_preds.append({
                    'boxes': p['boxes'].to(device),
                    'scores': p['scores'].to(device),
                    'labels': p['labels'].to(device)
                })

            if val_metric:
                val_metric.update(formatted_preds, formatted_targets)

        except Exception as e:
            print(f"\nError during validation inference or metric update: {e}")
            continue


    progress_bar_val.close()

    val_results = {}
    if val_metric:
        try:
            val_results = val_metric.compute()

            map_50 = val_results.get('map_50', torch.tensor(-1.0)).item()
            print(f"  Validation mAP@0.50: {map_50:.4f}")

        except Exception as e:
            print(f"Error computing validation metrics: {e}")

    return val_results


def train_retinanet_resnet50_sgd():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    dataset_train = get_voc_datasets(
        args.data_path, args.year, 'trainval',
        args.img_size
    )
    dataset_val = get_voc_datasets(
        args.data_path, args.year, 'test',
        args.img_size
    )

    if dataset_train is None or dataset_val is None or len(dataset_train) == 0 or len(dataset_val) == 0:
        print("Error: Could not load datasets or datasets are empty. Exiting.")
        return

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Dataset loading complete. Train: {len(dataset_train)} images, Val: {len(dataset_val)} images.")

    print("Loading model...")
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    model = retinanet_resnet50_fpn_v2(weights=weights)

    try:
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        original_norm_layer_type = type(model.head.classification_head.conv[0][1])

        if issubclass(original_norm_layer_type, nn.GroupNorm):
            num_groups = model.head.classification_head.conv[0][1].num_groups
            norm_layer_callable = functools.partial(nn.GroupNorm, num_groups)
        elif issubclass(original_norm_layer_type, nn.BatchNorm2d):
            norm_layer_callable = functools.partial(nn.GroupNorm, 32)
        else:
            norm_layer_callable = functools.partial(nn.GroupNorm, 32)

        new_cls_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=NUM_CLASSES,
            norm_layer=norm_layer_callable
        )
        model.head.classification_head = new_cls_head
    except Exception as e:
        print(f"Error modifying model head: {e}. Using default head for {NUM_CLASSES} classes.")
        model = retinanet_resnet50_fpn_v2(weights=weights, num_classes=NUM_CLASSES)


    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                if use_amp and 'amp_scaler_state_dict' in checkpoint and scaler:
                    scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
                start_epoch = checkpoint['epoch']
            except KeyError as e:
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0
        else:
            print(f"Checkpoint file not found at '{args.resume}'. Starting from scratch.")

    early_stopping = EarlyStopping(
        patience=args.patience,
        output_dir=args.output_dir,
        metric_name=args.early_stopping_metric,
        model_name_prefix=f"retinanet_resnet50_voc{args.year}"
    )

    print(f"Starting training from epoch {start_epoch + 1} to {args.epochs}")
    training_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler, use_amp, args)

        if "error" in train_metrics and train_metrics["error"] == "non-finite loss":
            print("Terminating training due to non-finite loss.")
            break

        run_validation = (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs
        val_results = {}
        if run_validation:
            val_results = evaluate(model, val_loader, device, epoch, args.epochs, args)


        epoch_duration = time.time() - start_time
        print(f"  Epoch Duration: {epoch_duration:.2f} seconds")

        if lr_scheduler:
            lr_scheduler.step()
            print(f"  Learning Rate: {lr_scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'epoch_{epoch+1}.pt')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                f'val_{early_stopping.metric_name}': val_results.get(early_stopping.metric_name, -1.0)
            }
            if lr_scheduler:
                 checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            if use_amp and scaler:
                 checkpoint['amp_scaler_state_dict'] = scaler.state_dict()
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                last_checkpoint_path = os.path.join(args.output_dir, 'last.pt')
                torch.save(checkpoint, last_checkpoint_path)
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        if run_validation:
            early_stopping(val_results, model, epoch, optimizer, lr_scheduler, args, scaler=scaler if use_amp else None)
            if early_stopping.early_stop:
                print("Early stopping triggered. Finishing training.")
                break


    total_training_time = time.time() - training_start_time
    print(f"\nTraining finished in {total_training_time / 3600:.2f} hours.")
    if early_stopping.best_score is not None:
        print(f"Best model saved to: {early_stopping.path} (Validation {early_stopping.metric_name}: {early_stopping.best_score:.4f})")
    else:
        print("Training finished, but no best model was saved (possibly due to no validation improvement or metrics error).")


def get_args():
    parser = argparse.ArgumentParser(description='Train RetinaNet ResNet50 FPN with SGD on VOC')
    parser.add_argument('--data-path', type=str, default='./VOCdevkit', help='Path to VOC dataset root (containing VOCdevkit)')
    parser.add_argument('--year', type=str, default='2007', help='VOC dataset year (e.g., 2007, 2012, or 2007+2012)')
    parser.add_argument('--img-size', type=int, default=320, help='Input image size (square images assumed)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate for SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay for SGD')
    parser.add_argument('--lr-step-size', type=int, default=50, help='Step size for LR scheduler (epochs)')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='Gamma for LR scheduler')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='outputs_retinanet_sgd_voc2007', help='Directory to save checkpoints and logs')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Path to latest checkpoint (default: none)')
    parser.add_argument('--patience', type=int, default=30, help='Patience for early stopping (epochs based on validation metric)')
    parser.add_argument('--early-stopping-metric', type=str, default='map_50', help='Metric to monitor for early stopping (e.g., map_50, map)')
    parser.add_argument('--val-interval', type=int, default=1, help='Run validation every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    return parser.parse_args()

if __name__ == "__main__":
    if os.name == 'posix':
        multiprocessing.set_start_method('spawn', force=True)
    elif os.name == 'nt':
        multiprocessing.freeze_support()

    train_retinanet_resnet50_sgd() 