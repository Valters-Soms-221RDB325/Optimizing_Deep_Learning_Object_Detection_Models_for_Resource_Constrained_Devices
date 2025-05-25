import torch
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
import functools
import numpy as np

try:
    import torchmetrics
    from torchmetrics.detection import MeanAveragePrecision
    torchmetrics_available = True
except ImportError:
    torchmetrics_available = False
    print("ERROR: torchmetrics not found. Evaluation cannot proceed without it.")
    print("Install with: pip install torchmetrics")
    exit(1)

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    fvcore_available = True
except ImportError:
    fvcore_available = False

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
        if target is None or 'annotation' not in target:
            return image, {'boxes': tv_tensors.BoundingBoxes(torch.empty((0, 4), dtype=torch.float32), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=image.shape[-2:]), 'labels': torch.empty(0, dtype=torch.int64)}

        try:
            annotation = target['annotation']
            img_width = int(annotation['size']['width'])
            img_height = int(annotation['size']['height'])
            objects = annotation.get('object', [])
        except (KeyError, TypeError) as e:
             return image, {'boxes': tv_tensors.BoundingBoxes(torch.empty((0, 4), dtype=torch.float32), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=image.shape[-2:]), 'labels': torch.empty(0, dtype=torch.int64)}

        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        labels = []

        for obj in objects:
            is_difficult = int(obj.get('difficult', '0')) == 1
            if is_difficult:
                continue

            class_name = obj.get('name')
            if class_name not in self.class_to_idx:
                continue

            bbox = obj.get('bndbox')
            if bbox is None:
                continue

            try:
                xmin = max(0.0, float(bbox['xmin']))
                ymin = max(0.0, float(bbox['ymin']))
                xmax = min(float(img_width), float(bbox['xmax']))
                ymax = min(float(img_height), float(bbox['ymax']))
                box = [xmin, ymin, xmax, ymax]

            except (KeyError, ValueError, TypeError) as e:
                 continue

            if box[2] <= box[0] or box[3] <= box[1]:
                 continue

            boxes.append(box)
            labels.append(self.class_to_idx[class_name])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty(0, dtype=torch.int64)

        target_dict = {
            'boxes': tv_tensors.BoundingBoxes(
                boxes_tensor,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=image.shape[-2:]),
            'labels': labels_tensor
        }
        return image, target_dict

class InitialTransform:
    def __init__(self, class_to_idx):
        self.target_transform = TargetTransformV2(class_to_idx=class_to_idx)

    def __call__(self, img, target):
        img_tensor = transforms.functional.to_image(img)
        img_tensor_out, target_dict_out = self.target_transform(img_tensor, target)
        return img_tensor_out, target_dict_out

def get_eval_args():
    parser = argparse.ArgumentParser(description='Evaluate RetinaNet ResNet50 FPN V2 on VOC 2007 Test Set')
    parser.add_argument('--data-path', type=str, default='./datasets/VOC/VOCdevkit',
                        help='Root path to VOCdevkit directory')
    parser.add_argument('--year', type=str, default='2007', help='VOC dataset year (e.g., 2007)')
    parser.add_argument('--img-size', type=int, default=320,
                        help='Input image size used during training (e.g., 320, 600, 800)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation (adjust based on GPU memory)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--checkpoint', required=True, type=str, metavar='PATH',
                        help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for consistency if needed)')
    parser.add_argument('--measure-latency', action='store_true', help='Measure and report inference latency')
    parser.add_argument('--latency-warmup', type=int, default=10, help='Number of warmup batches for latency measurement')
    parser.add_argument('--latency-measure', type=int, default=50, help='Number of measured batches for latency measurement')

    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None
    return tuple(zip(*batch))

def get_voc_test_dataset(data_path, year, img_size):
    voc_devkit_path = os.path.join(data_path, 'VOCdevkit')
    if not os.path.isdir(voc_devkit_path):
        voc_year_path = os.path.join(voc_devkit_path, f'VOC{year}')
        if not os.path.isdir(voc_year_path):
             raise FileNotFoundError(f"VOCdevkit/VOC{year} not found at expected path: {voc_year_path}")

    initial_transformer = InitialTransform(class_to_idx=CLASS_TO_IDX)

    eval_transforms = transforms.Compose([
        initial_transformer,
        transforms.ToDtype(torch.float32, scale=False),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.SanitizeBoundingBoxes(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    try:
        dataset = VOCDetection(
            root=data_path,
            year=year,
            image_set='test',
            download=False,
            transforms=eval_transforms
        )
        if not dataset or len(dataset) == 0:
             raise ValueError(f"Dataset loaded successfully but is empty for image_set='test'. Check VOC files.")

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error loading VOC dataset: {e}")
        print(f"Checked for VOCdevkit in: '{data_path}'")
        print(f"Please ensure the VOC {year} dataset (including test set) is correctly placed.")
        exit(1)

    return dataset

def evaluate_retinanet():
    args = get_eval_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_test = get_voc_test_dataset(
        args.data_path, args.year, args.img_size
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )

    model = retinanet_resnet50_fpn_v2(weights=None, num_classes=NUM_CLASSES)

    try:
        _temp_model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        in_channels = _temp_model.head.classification_head.conv[0][0].in_channels
        num_anchors = _temp_model.anchor_generator.num_anchors_per_location()[0]
        original_norm_layer_type = type(_temp_model.head.classification_head.conv[0][1])
        del _temp_model

        if issubclass(original_norm_layer_type, nn.GroupNorm):
            num_groups = 32
            norm_layer_callable = functools.partial(nn.GroupNorm, num_groups)
        else:
            norm_layer_callable = functools.partial(nn.GroupNorm, 32)

        new_cls_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=NUM_CLASSES,
            norm_layer=norm_layer_callable
        )
        model.head.classification_head = new_cls_head

    except NameError:
        print("Error: RetinaNet_ResNet50_FPN_V2_Weights not defined. Ensure torchvision is updated and import is correct.")
        print("Skipping automatic head configuration inspection. Ensure model matches checkpoint.")
    except Exception as e:
        print(f"Error inspecting model head structure or replacing head: {e}")
        print("Could not automatically determine parameters for head replacement. Ensure the model architecture matches the checkpoint.")

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
        except TypeError as e:
             try:
                 sample_input_list = [torch.randn(1, 3, args.img_size, args.img_size).to(device)]
                 flop_analyzer = FlopCountAnalysis(model, sample_input_list)
                 total_flops = flop_analyzer.total()
                 gflops = total_flops / 1e9
             except Exception as e_list:
                 print(f"FLOPs calculation with list input also failed: {e_list}")
                 gflops = -1.0
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            gflops = -1.0

    avg_latency_ms = -1.0
    if args.measure_latency:
        latencies = []
        try:
            latency_loader_iter = iter(test_loader)
            sample_images, _ = next(latency_loader_iter)
            sample_images = list(image.to(device) for image in sample_images)

            if not sample_images:
                 raise ValueError("Failed to get a valid sample batch for latency test.")

            with torch.no_grad():
                for _ in range(args.latency_warmup):
                    _ = model(sample_images)

            with torch.no_grad():
                for i in tqdm(range(args.latency_measure), desc="Latency Benchmark", leave=False):
                    if device.type == 'cuda':
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        start_event.record()
                        _ = model(sample_images)
                        end_event.record()
                        torch.cuda.synchronize()
                        batch_latency_ms = start_event.elapsed_time(end_event)
                    else:
                        start_time = time.perf_counter()
                        _ = model(sample_images)
                        end_time = time.perf_counter()
                        batch_latency_ms = (end_time - start_time) * 1000

                    current_batch_size = len(sample_images)
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

    with torch.no_grad():
        for batch_data in progress_bar_eval:
            if batch_data is None or batch_data[0] is None:
                continue
            try:
                images, targets = batch_data
            except ValueError:
                continue

            valid_indices = [idx for idx, (img, tgt) in enumerate(zip(images, targets)) if img is not None and tgt is not None]
            if not valid_indices:
                continue
            images = [images[idx] for idx in valid_indices]
            targets = [targets[idx] for idx in valid_indices]

            if not images:
                continue

            images = list(image.to(device) for image in images)

            formatted_targets = []
            for t in targets:
                if not isinstance(t, dict) or 'boxes' not in t or 'labels' not in t:
                    continue

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

            if not formatted_targets:
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
                val_metric.update(formatted_preds, formatted_targets)
            except Exception as e:
                print(f"\nError during model inference or mAP metric update: {e}")
                print("Stopping evaluation due to error.")
                return

    progress_bar_eval.close()

    try:
        results = val_metric.compute()

        map_val = results.get('map', torch.tensor(-1.0)).item()
        map_50 = results.get('map_50', torch.tensor(-1.0)).item()
        map_75 = results.get('map_75', torch.tensor(-1.0)).item()
        map_small = results.get('map_small', torch.tensor(-1.0)).item()
        map_medium = results.get('map_medium', torch.tensor(-1.0)).item()
        map_large = results.get('map_large', torch.tensor(-1.0)).item()

        print("\n--- Evaluation Summary ---")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Device: {device.type}")
        print(f"  Image Size: {args.img_size}x{args.img_size}")
        if gflops > 0:
            print(f"  GFLOPs (batch size 1): {gflops:.2f}")
        else:
            print(f"  GFLOPs: Not calculated")
        if avg_latency_ms > 0:
            print(f"  Avg. Latency per image: {avg_latency_ms:.2f} ms (Eval Batch Size: {args.batch_size})")
        else:
            print(f"  Avg. Latency per image: Not calculated")
        print("\n--- mAP Results ---")
        print(f"  mAP (IoU=0.50:0.95): {map_val:.4f}")
        print(f"  mAP (IoU=0.50):      {map_50:.4f}")
        print(f"  mAP (IoU=0.75):      {map_75:.4f}")
        print("---")
        print(f"  mAP (small areas):   {map_small:.4f}")
        print(f"  mAP (medium areas):  {map_medium:.4f}")
        print(f"  mAP (large areas):   {map_large:.4f}")
        print("------------------------")

        per_class_metric_key = 'map_50_per_class'
        if per_class_metric_key not in results:
             per_class_metric_key = 'map_per_class'

        if per_class_metric_key in results:
             class_aps_50 = results[per_class_metric_key]
             print("\n--- AP@0.50 per class ---")
             if class_aps_50 is not None and len(class_aps_50) == len(VOC_CLASSES):
                 for i, class_name in enumerate(VOC_CLASSES):
                     print(f"  {class_name:<15}: {class_aps_50[i].item():.4f}")
             else:
                 print(f"  Warning: Per-class AP data format issue. Expected {len(VOC_CLASSES)} values, got: {class_aps_50}")
        else:
            print("\nPer-class AP data not found in results.")

    except Exception as e:
        print(f"Error computing or printing final metrics: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    evaluate_retinanet()