import torch
import torchvision
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
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
import functools
import numpy as np
import random

try:
    import torchmetrics
    from torchmetrics.detection import MeanAveragePrecision
    torchmetrics_available = True
except ImportError:
    torchmetrics_available = False
    print("ERROR: torchmetrics not found. mAP calculation cannot proceed.")
    print("Install with: pip install torchmetrics")
    exit(1)

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    fvcore_available = True
except ImportError:
    fvcore_available = False
    print("WARNING: fvcore not found. FLOPs calculation will be skipped.")
    print("Install with: pip install fvcore")

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_IDX = {cls_name: i + 1 for i, cls_name in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES) + 1

class TargetDictToTensorDict:
    def __init__(self, class_to_idx):
        self.class_to_idx = class_to_idx
        self.label_map = {name: idx for name, idx in class_to_idx.items()}
        self.label_map['background'] = 0

    def __call__(self, target_dict, canvas_size):
        height, width = canvas_size
        boxes = []
        labels = []
        image_id = target_dict.get('annotation', {}).get('filename', 'unknown')
        size_info = target_dict.get('annotation', {}).get('size', {})
        orig_width = int(size_info.get('width', width))
        orig_height = int(size_info.get('height', height))

        objects = target_dict.get('annotation', {}).get('object', [])
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            label_name = obj.get('name')
            if label_name not in self.label_map:
                continue

            bbox = obj.get('bndbox')
            if not bbox:
                continue

            try:
                xmin = max(0, int(float(bbox['xmin'])))
                ymin = max(0, int(float(bbox['ymin'])))
                xmax = min(orig_width, int(float(bbox['xmax'])))
                ymax = min(orig_height, int(float(bbox['ymax'])))

                if xmin >= xmax or ymin >= ymax:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.label_map[label_name])
            except (KeyError, ValueError) as e:
                continue

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        return {
            "boxes": tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=canvas_size),
            "labels": labels_tensor
        }

class ApplyTargetConversion:
    def __init__(self, target_converter):
        self.target_converter = target_converter

    def __call__(self, img, target_dict):
        canvas_size = (img.height, img.width)
        tv_target = self.target_converter(target_dict, canvas_size)
        return img, tv_target

def collate_fn(batch):
    images = []
    targets = []
    for item in batch:
        if isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
            images.append(item[0])
            targets.append(item[1])
        else:
            continue

    if not images:
        return [], []

    try:
        images = torch.stack(images, 0)
    except RuntimeError as e:
         print(f"Error stacking images in collate_fn: {e}. Likely inconsistent shapes.")
         for i, img_tensor in enumerate(images):
             print(f"  Image {i} shape: {img_tensor.shape}")
         return [], []
    except Exception as e:
         print(f"Unexpected error stacking images: {e}")
         return [], []

    processed_targets = []
    for target in targets:
        if "boxes" in target and isinstance(target["boxes"], tv_tensors.BoundingBoxes):
            boxes = target["boxes"]
            labels = target["labels"]

            if boxes.numel() == 0:
                processed_targets.append(target)
                continue
            try:
                w = boxes.data[:, 2] - boxes.data[:, 0]
                h = boxes.data[:, 3] - boxes.data[:, 1]
                valid_boxes_mask = (w > 0) & (h > 0)

                filtered_boxes = boxes[valid_boxes_mask]
                filtered_labels = labels[valid_boxes_mask]

                new_target = {k: v for k, v in target.items() if k not in ["boxes", "labels"]}
                new_target["boxes"] = filtered_boxes
                new_target["labels"] = filtered_labels
                processed_targets.append(new_target)
            except IndexError as e:
                processed_targets.append(target)
            except Exception as e:
                processed_targets.append(target)

        else:
            processed_targets.append(target)

    targets = processed_targets
    return images, targets

def get_eval_args():
    parser = argparse.ArgumentParser(description='Evaluate RetinaNet ShuffleNetV2 FPN on VOC 2007 Test Set')
    parser.add_argument('--data-path', type=str, required=True, help='Path to VOC dataset root (containing VOCdevkit)')
    parser.add_argument('--year', type=str, default='2007', help='VOC dataset year (e.g., 2007)')
    parser.add_argument('--img-size', type=int, default=320, help='Input image size used during training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint', required=True, type=str, metavar='PATH', help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--measure-latency', action='store_true', help='Measure and report inference latency')
    parser.add_argument('--latency-warmup', type=int, default=10, help='Warmup batches for latency measurement')
    parser.add_argument('--latency-measure', type=int, default=50, help='Measured batches for latency measurement')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_voc_test_dataset(data_path, year, img_size):
    """Loads the VOC test dataset with evaluation transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    target_converter = TargetDictToTensorDict(CLASS_TO_IDX)

    eval_transforms = transforms.Compose([
        ApplyTargetConversion(target_converter),
        transforms.ToImage(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ])

    expected_voc_path = os.path.join(data_path, "VOCdevkit", f"VOC{year}")
    if not os.path.isdir(expected_voc_path):
        raise FileNotFoundError(f"VOC dataset not found at '{expected_voc_path}'.")
    voc_root_for_dataset = data_path

    dataset = VOCDetection(
        root=voc_root_for_dataset,
        year=year,
        image_set='test',
        download=False,
        transforms=eval_transforms,
        transform=None,
        target_transform=None
    )
    if not dataset or len(dataset) == 0:
        raise ValueError("Dataset loaded successfully but is empty for image_set='test'.")
    return dataset

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

# --- Main Evaluation Function ---
def evaluate_retinanet_shufflenet():
    args = get_eval_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_test = get_voc_test_dataset(args.data_path, args.year, args.img_size)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True if device == 'cuda' else False)

    backbone_net = shufflenet_v2_x1_0(weights=None)
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

    backbone = ShuffleNetFPNBackbone(body, fpn, fpn_out_channels)

    anchor_generator = AnchorGenerator(
        sizes=((32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = RetinaNet(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        anchor_generator=anchor_generator,
    )

    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint file not found at '{args.checkpoint}'")
        return

    print(f"Loading checkpoint: '{args.checkpoint}'")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if 'model_state_dict' not in checkpoint:
             raise KeyError("Checkpoint does not contain 'model_state_dict'.")

        load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    except KeyError as e:
         print(f"Error: Checkpoint file '{args.checkpoint}' is missing expected key: {e}.")
         return
    except RuntimeError as e:
        print(f"Error: State dictionary does not match model architecture: {e}")
        print("Ensure the model definition exactly matches the training script.")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    gflops = -1.0
    if fvcore_available:
        try:
            sample_input = (torch.randn(1, 3, args.img_size, args.img_size).to(device),)
            flop_analyzer = FlopCountAnalysis(model, sample_input)
            total_flops = flop_analyzer.total()
            gflops = total_flops / 1e9
        except Exception as e:
            try:
                 sample_input_list = [torch.randn(1, 3, args.img_size, args.img_size).to(device)]
                 flop_analyzer = FlopCountAnalysis(model, sample_input_list)
                 total_flops = flop_analyzer.total()
                 gflops = total_flops / 1e9
            except Exception as e_list:
                 print(f"FLOPs calculation with list input also failed: {e_list}")
                 gflops = -1.0
    else:
        print("fvcore not available, skipping FLOPs calculation.")

    avg_latency_ms = -1.0
    if args.measure_latency:
        latencies = []
        try:
            latency_loader_iter = iter(test_loader)
            initial_images, _ = next(latency_loader_iter)
            initial_images = list(image.to(device) for image in initial_images)
            if not initial_images:
                 raise ValueError("Failed to get a valid sample batch for latency test.")
            actual_batch_size = len(initial_images)

            with torch.no_grad():
                for _ in range(args.latency_warmup):
                    _ = model(initial_images)

            with torch.no_grad():
                for i in tqdm(range(args.latency_measure), desc="Latency Benchmark", leave=False):
                    measure_images = initial_images
                    if device.type == 'cuda':
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        start_event.record()
                        _ = model(measure_images)
                        end_event.record()
                        torch.cuda.synchronize()
                        batch_latency_ms = start_event.elapsed_time(end_event)
                    else:
                        start_time = time.perf_counter()
                        _ = model(measure_images)
                        end_time = time.perf_counter()
                        batch_latency_ms = (end_time - start_time) * 1000

                    current_batch_size = len(measure_images)
                    if current_batch_size > 0:
                        latencies.append(batch_latency_ms / current_batch_size)

            if latencies:
                avg_latency_ms = sum(latencies) / len(latencies)
            else:
                avg_latency_ms = -1.0

        except StopIteration:
             if latencies:
                 avg_latency_ms = sum(latencies) / len(latencies)
             else:
                 avg_latency_ms = -1.0
        except Exception as e:
            print(f"Error during latency measurement: {e}")
            avg_latency_ms = -1.0

    val_metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=torch.linspace(0.5, 0.95, 10).tolist(),
        class_metrics=True
    )
    val_metric.to(device)

    progress_bar_eval = tqdm(test_loader, desc="Evaluating mAP", leave=False)
    eval_start_time = time.time()

    with torch.no_grad():
        for batch_data in progress_bar_eval:
            if not batch_data or not batch_data[0].numel():
                 continue
            try:
                images, targets = batch_data
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            except Exception as e:
                 continue

            try:
                predictions = model(images)

                formatted_preds = []
                for p in predictions:
                    formatted_preds.append({
                        'boxes': p['boxes'].to(device),
                        'scores': p['scores'].to(device),
                        'labels': p['labels'].to(device)
                    })
                formatted_targets = []
                for t in targets:
                     boxes_tensor = t.get('boxes')
                     labels_tensor = t.get('labels')
                     if isinstance(boxes_tensor, torch.Tensor) and isinstance(labels_tensor, torch.Tensor):
                        formatted_targets.append({
                           'boxes': boxes_tensor.to(device),
                           'labels': labels_tensor.to(device)
                        })

                if formatted_preds and formatted_targets:
                     val_metric.update(formatted_preds, formatted_targets)

            except Exception as e:
                print(f"\nError during model inference or mAP metric update: {e}")
                import traceback
                traceback.print_exc()
                print("Stopping evaluation due to error.")
                return

    progress_bar_eval.close()
    eval_duration = time.time() - eval_start_time

    try:
        results = val_metric.compute()

        map_val = results.get('map', torch.tensor(-1.0)).item()
        map_50 = results.get('map_50', torch.tensor(-1.0)).item()
        map_75 = results.get('map_75', torch.tensor(-1.0)).item()
        map_small = results.get('map_small', torch.tensor(-1.0)).item()
        map_medium = results.get('map_medium', torch.tensor(-1.0)).item()
        map_large = results.get('map_large', torch.tensor(-1.0)).item()

        print("\n--- Evaluation Summary (ShuffleNetV2 Backbone) ---")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Device: {device.type}")
        print(f"  Image Size: {args.img_size}x{args.img_size}")
        if gflops > 0: print(f"  GFLOPs (batch size 1): {gflops:.2f}")
        else: print(f"  GFLOPs: Not calculated")
        if avg_latency_ms > 0: print(f"  Avg. Latency per image: {avg_latency_ms:.2f} ms (Eval Batch Size: {args.batch_size})")
        else: print(f"  Avg. Latency per image: Not calculated")
        print("\n--- mAP Results ---")
        print(f"  mAP (IoU=0.50:0.95): {map_val:.4f}")
        print(f"  mAP (IoU=0.50):      {map_50:.4f}")
        print(f"  mAP (IoU=0.75):      {map_75:.4f}")
        print("---")
        print(f"  mAP (small areas):   {map_small:.4f}")
        print(f"  mAP (medium areas):  {map_medium:.4f}")
        print(f"  mAP (large areas):   {map_large:.4f}")
        print("------------------------")

        per_class_metric_key = 'map_per_class'
        map_50_per_class_key = 'map_50_per_class'

        if map_50_per_class_key in results:
             class_aps_50 = results[map_50_per_class_key]
             print("\n--- AP@0.50 per class ---")
        elif per_class_metric_key in results:
             class_aps_50 = results[per_class_metric_key]
             print("\n--- AP@0.50:0.95 per class (fallback) ---")
        else:
            class_aps_50 = None
            print("\nPer-class AP data not found in results.")


        if class_aps_50 is not None:
             if isinstance(class_aps_50, torch.Tensor) and class_aps_50.numel() == len(VOC_CLASSES):
                 for i, class_name in enumerate(VOC_CLASSES):
                     print(f"  {class_name:<15}: {class_aps_50[i].item():.4f}")
             else:
                 print(f"  Warning: Per-class AP data format issue. Expected tensor with {len(VOC_CLASSES)} values, got: {type(class_aps_50)}, Data: {class_aps_50}")


    except Exception as e:
        print(f"Error computing or printing final metrics: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    evaluate_retinanet_shufflenet()

# --- END OF FILE evaluate_retinanet_shufflenet.py ---