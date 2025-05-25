import torch
import torch.optim as optim
import torchvision
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import os
import argparse
from tqdm import tqdm
import time
import warnings
import multiprocessing
import torch.nn as nn
import torchvision.tv_tensors as tv_tensors
import random
import numpy as np
import functools

try:
    import torchmetrics
    from torchmetrics.detection import MeanAveragePrecision
    torchmetrics_available = True
except ImportError:
    torchmetrics_available = False
    print("WARNING: torchmetrics not found. Validation mAP calculation will be skipped.")
    print("Install with: pip install torchmetrics pycocotools")


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_IDX = {cls_name: i + 1 for i, cls_name in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES) + 1

class TargetTransformV2:
    def __init__(self, class_to_idx):
        self.class_to_idx = class_to_idx

    def __call__(self, image, target):
        if target is None or 'annotation' not in target or 'object' not in target['annotation']:
            warnings.warn(f"Target or annotation or object not found in target: {target}. Returning empty labels/boxes.")
            return image, {'boxes': tv_tensors.BoundingBoxes(torch.empty((0, 4), dtype=torch.float32), format="XYXY", canvas_size=image.shape[-2:]),
                           'labels': torch.empty((0,), dtype=torch.int64)}

        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        labels = []
        for obj in objects:
            try:
                label_name = obj['name']
                if label_name not in self.class_to_idx:
                    warnings.warn(f"Unknown class name '{label_name}' in image {target['annotation'].get('filename', 'UnknownFile')}. Skipping object.")
                    continue
                labels.append(self.class_to_idx[label_name])

                bndbox = obj['bndbox']
                xmin = float(bndbox['xmin'])
                ymin = float(bndbox['ymin'])
                xmax = float(bndbox['xmax'])
                ymax = float(bndbox['ymax'])
                boxes.append([xmin, ymin, xmax, ymax])
            except KeyError as e:
                warnings.warn(f"Missing key {e} in object {obj} for image {target['annotation'].get('filename', 'UnknownFile')}. Skipping object.")
                continue
            except ValueError as e:
                warnings.warn(f"ValueError parsing bndbox for object {obj} in image {target['annotation'].get('filename', 'UnknownFile')}: {e}. Skipping object.")
                continue


        if not boxes:
            return image, {'boxes': tv_tensors.BoundingBoxes(torch.empty((0, 4), dtype=torch.float32), format="XYXY", canvas_size=image.shape[-2:]),
                           'labels': torch.empty((0,), dtype=torch.int64)}

        boxes_tensor = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=image.shape[-2:])
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        return image, {'boxes': boxes_tensor, 'labels': labels_tensor}

def get_voc_datasets(data_path, year, image_set, img_size):
    voc_root = Path(data_path)
    if not voc_root.exists():
        raise FileNotFoundError(f"VOC dataset root not found at {data_path}")
    if not (voc_root / "VOCdevkit").exists():
         raise FileNotFoundError(f"VOCdevkit not found in {data_path}. Please ensure {data_path} is the parent directory of VOCdevkit.")

    target_transform = TargetTransformV2(CLASS_TO_IDX)

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.SanitizeBoundingBoxes(),
        transforms.Lambda(target_transform)
    ])

    try:
        dataset = VOCDetection(
            root=str(voc_root),
            year=year,
            image_set=image_set,
            download=False,
            transforms=transform
        )
        if len(dataset) == 0:
            warnings.warn(f"VOC dataset for year {year}, image_set '{image_set}' is empty. Check paths and dataset integrity.")
    except Exception as e:
        print(f"Error loading VOC dataset: {e}")
        print(f"  Path: {data_path}, Year: {year}, Image Set: {image_set}")
        print("  Please ensure the VOC dataset is correctly structured and accessible.")
        raise
    return dataset

def collate_fn(batch):
    images = []
    targets = []
    for item in batch:
        image, target = item
        if image is not None and target is not None:
            images.append(image)
            targets.append(target)
        else:
            warnings.warn("Skipping a None item in batch during collation.")

    if not images:
        return None, None

    return images, targets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, metric_name='val_loss'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.metric_name = metric_name
        if verbose:
            self.trace_func(f"EarlyStopping initialized: patience={patience}, metric='{metric_name}', path='{path}'")

    def __call__(self, metrics, model, epoch, optimizer, lr_scheduler, args, scaler=None):
        if not metrics or self.metric_name not in metrics:
            if self.verbose:
                self.trace_func(f"EarlyStopping: Metric '{self.metric_name}' not found in validation results. Skipping check.")
            return

        score = metrics[self.metric_name]

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, epoch, optimizer, lr_scheduler, args, scaler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best {self.metric_name}: {self.best_score:.4f}, Current: {score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, epoch, optimizer, lr_scheduler, args, scaler)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, epoch, optimizer, lr_scheduler, args, scaler):
        if self.verbose:
            self.trace_func(f"Validation {self.metric_name} improved ({self.val_metric_min:.4f} --> {val_metric:.4f}). Saving model to {self.path} ...")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            f'val_{self.metric_name}': val_metric
        }
        if lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        if scaler:
            checkpoint['amp_scaler_state_dict'] = scaler.state_dict()
        try:
            torch.save(checkpoint, self.path)
        except Exception as e:
            self.trace_func(f"Error saving best model checkpoint: {e}")
        self.val_metric_min = val_metric


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, use_amp, print_freq=10):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch+1}]"
    lr_scheduler = None

    optimizer.zero_grad()

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not np.isfinite(loss_value):
            print(f"WARNING: Non-finite loss detected: {loss_value}. Skipping batch.")
            optimizer.zero_grad()
            return {"error": "non-finite loss"}


        if use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        optimizer.zero_grad()

        metric_logger.update(loss=loss_value, **{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, torchmetrics_available, num_classes):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test Epoch: [{epoch+1}]"
    results_list = []

    if torchmetrics_available:
        map_metric = MeanAveragePrecision(class_metrics=True, iou_type="bbox")
        map_metric.to(device)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [img.to(device) for img in images]
        original_image_sizes = [img.shape[-2:] for img in images]

        if device.type == 'cuda':
            torch.cuda.synchronize()
        model_time = time.time()

        with torch.cuda.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(images)

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        if torchmetrics_available:
            formatted_targets = []
            for t_idx, t in enumerate(targets):
                boxes_xyxy = t['boxes']
                if not isinstance(boxes_xyxy, tv_tensors.BoundingBoxes):
                     boxes_xyxy = tv_tensors.BoundingBoxes(boxes_xyxy, format="XYXY", canvas_size=original_image_sizes[t_idx])
                else:
                    boxes_xyxy = boxes_xyxy.to('cpu')


                formatted_targets.append({
                    'boxes': boxes_xyxy.data,
                    'labels': t['labels'].to('cpu', dtype=torch.int)
                })
            map_metric.update(outputs, formatted_targets)

        metric_logger.update(model_time=model_time)

    metric_logger.synchronize_between_processes()

    if torchmetrics_available:
        try:
            computed_metrics = map_metric.compute()
            return computed_metrics
        except Exception as e:
            print(f"Error computing mAP with torchmetrics: {e}")
            return {}
    return {}


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].add_value(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.add_value(time.time() - end)
            yield obj
            iter_time.add_value(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(time.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(time.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = मध्यिका = [] # Using list as deque
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.total += value * n
        self.count += n

    def synchronize_between_processes(self):
        pass

    def add_value(self, value):
        self.update(value)

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else float('nan')

    @property
    def max(self):
        return max(self.deque) if self.deque else float('-inf')

    @property
    def value(self):
        return self.deque[-1] if self.deque else float('nan')

    def __str__(self):
        if not self.deque:
            return self.fmt.format(
                median=float('nan'),
                avg=float('nan'),
                global_avg=float('nan'),
                max=float('-inf'),
                value=float('nan'))
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )

def train_retinanet_shufflenet_sgd():
    args = get_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, AMP: {device.type == 'cuda'}")

    os.makedirs(args.output_dir, exist_ok=True)

    scaler = torch.amp.GradScaler(device=device, enabled=(device.type == 'cuda'))

    dataset_train = get_voc_datasets(args.data_path, args.year, 'trainval', args.img_size)
    dataset_val = get_voc_datasets(args.data_path, args.year, 'test', args.img_size)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=(device.type == 'cuda'), drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=(device.type == 'cuda'))
    print(f"Datasets loaded. Train: {len(dataset_train)}, Val: {len(dataset_val)} images.")

    backbone_net = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
    backbone_net.fc = nn.Identity()

    return_layers = {'stage2': '0', 'stage3': '1', 'stage4': '2'}
    in_channels_list = [116, 232, 464]
    fpn_out_channels = 256
    fpn_extra_blocks = LastLevelP6P7(fpn_out_channels, fpn_out_channels)
    body = IntermediateLayerGetter(backbone_net, return_layers=return_layers)
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=fpn_out_channels,
        extra_blocks=fpn_extra_blocks,
    )

    class ShuffleNetFPNBackbone(nn.Module):
        def __init__(self, body, fpn, fpn_output_channels):
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.out_channels = fpn_output_channels

        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)
            return x

    backbone = ShuffleNetFPNBackbone(body, fpn, fpn_out_channels)

    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = RetinaNet(
        backbone,
        num_classes=NUM_CLASSES,
        anchor_generator=anchor_generator,
    )
    model.to(device)
    print(f"Model RetinaNet+ShuffleNetV2 loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print(f"Optimizer: SGD, LR: {args.lr}, Momentum: {args.momentum}, WD: {args.weight_decay}")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                if 'amp_scaler_state_dict' in checkpoint and (device.type == 'cuda'):
                    scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            except KeyError:
                 print("Checkpoint is missing expected keys (e.g. 'model_state_dict', 'optimizer_state_dict', 'epoch'). Trying to load only model_state_dict if it's a raw state_dict.")
                 try:
                     model.load_state_dict(checkpoint) # Attempt to load if it's just a state_dict
                     print("Successfully loaded model_state_dict from raw checkpoint. Optimizer and epoch not resumed.")
                 except Exception as e_raw:
                     print(f"Could not load raw state_dict: {e_raw}")
                     print("Could not load model from checkpoint due to missing keys or incompatible structure.")
            except Exception as e:
                print(f"An unexpected error occurred while loading checkpoint: {e}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")


    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(args.output_dir, 'best_model_shufflenet.pt'), metric_name='map_50')

    training_start_time = time.time()
    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler, (device.type == 'cuda'), print_freq=50)

        if "error" in train_metrics and train_metrics["error"] == "non-finite loss":
            print("Terminating training due to non-finite loss.")
            break

        val_results = {}
        if (epoch + 1) % 1 == 0 or (epoch + 1) == total_epochs :
            if torchmetrics_available:
                try:
                    val_results = evaluate(model, val_loader, device, epoch, torchmetrics_available, NUM_CLASSES)
                    print(f"Epoch {epoch+1}/{total_epochs} - Val mAP@0.50: {val_results.get('map_50', -1.0):.4f}, mAP@0.50:0.95: {val_results.get('map', -1.0):.4f}")
                except Exception as compute_e:
                    print(f"Error computing validation metric: {compute_e}")
                    val_results = {}
            else:
                print(f"Epoch {epoch+1}/{total_epochs} - Validation metrics skipped (torchmetrics not available).")


        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch Duration: {epoch_duration:.2f}s. LR: {optimizer.param_groups[0]['lr']:.6f}")

        if lr_scheduler:
            lr_scheduler.step()

        if (epoch + 1) % args.save_period == 0 or (epoch + 1) == total_epochs:
            checkpoint_path = os.path.join(args.output_dir, f'epoch_{epoch+1}_shufflenet.pt')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                f'val_{early_stopping.metric_name}': val_results.get(early_stopping.metric_name, -1.0)
            }
            if lr_scheduler:
                 checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            if (device.type == 'cuda'):
                 checkpoint['amp_scaler_state_dict'] = scaler.state_dict()
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                last_checkpoint_path = os.path.join(args.output_dir, 'last_shufflenet.pt')
                torch.save(checkpoint, last_checkpoint_path)
            except Exception as e:
                print(f"Error saving checkpoint: {e}")


        early_stopping(val_results, model, epoch, optimizer, lr_scheduler, args, scaler=scaler if (device.type == 'cuda') else None)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    total_training_time = time.time() - training_start_time
    print(f"\nTraining finished in {total_training_time / 3600:.2f} hours.")
    if early_stopping.best_score is not None:
        print(f"Best model saved to: {early_stopping.path} (Validation {early_stopping.metric_name}: {early_stopping.best_score:.4f})")
    else:
        print("Training finished, but no best model was saved (possibly due to no validation improvement or metrics error).")


def get_parser():
    parser = argparse.ArgumentParser(description='Train RetinaNet ShuffleNetV2 FPN with SGD on VOC')
    parser.add_argument('--data-path', type=str, required=True, help='Path to VOC dataset root (containing VOCdevkit)')
    parser.add_argument('--year', type=str, default='2007', help='VOC dataset year (e.g., 2007)')
    parser.add_argument('--img-size', type=int, default=320, help='Input image size (e.g., 320)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training (adjust based on GPU memory)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate for SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay for SGD')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='outputs_retinanet_shufflenet', help='Directory to save checkpoints and logs')
    parser.add_argument('--save-period', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Path to latest checkpoint (default: none)')
    parser.add_argument('--patience', type=int, default=200, help='Patience for early stopping (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    return parser

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_retinanet_shufflenet_sgd() 