import torch
import torchvision
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.datasets import VOCDetection
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteHead, SSDLiteClassificationHead
from torchvision.ops.misc import Conv2dNormActivation
import torch.nn as nn

import time
import math
import sys
import datetime
import os
import traceback
import xml.etree.ElementTree as ET
from PIL import Image
import warnings

try:
    import torchvision.tv_tensors as tv_tensors
except ImportError:
    tv_tensors = None

VOC_ROOT_DIR = "C:/SSD_folder/VOC"
YEAR = '2007'
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 32
NUM_WORKERS = 8
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 200
PRINT_FREQ = 50
MODEL_SAVE_PATH = "C:/SSD_folder/checkpoints/ckpt_epoch_189.pth"

CHECKPOINT_DIR = "checkpoints"
LATEST_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "latest_ckpt.pth")
EPOCH_CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt_epoch_")
CHECKPOINT_SAVE_FREQ = 10

def get_object_detection_model():
    """
    Loads the pre-trained SSDLite MobileNetV3 Large model and modifies
    its classification head for the number of classes specified in config.py.
    Extracts necessary parameters by inspecting the existing model structure.
    """
    print("Loading pre-trained SSDLite320_MobileNet_V3_Large model...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    print("Pre-trained model loaded.")

    try:
        num_anchors = model.anchor_generator.num_anchors_per_location()
        if not isinstance(num_anchors, (list, tuple)) or not all(isinstance(n, int) for n in num_anchors):
             raise TypeError(f"num_anchors_per_location did not return list/tuple of ints: {num_anchors}")
        print(f"Retrieved num_anchors per location: {num_anchors}")
        if len(num_anchors) != 6:
             print(f"Warning: Expected 6 anchor values, got {len(num_anchors)}. May mismatch head.")

    except Exception as e:
        print(f"ERROR: Could not get num_anchors_per_location: {e}"); raise

    head_in_channels = []
    try:
        existing_head = model.head
        if not isinstance(existing_head, SSDLiteHead): raise TypeError(f"Model head is not SSDLiteHead")
        if not hasattr(existing_head, 'classification_head'): raise AttributeError("Head has no 'classification_head'")
        if not hasattr(existing_head.classification_head, 'module_list'): raise AttributeError("classification_head has no 'module_list'")

        for seq_module in existing_head.classification_head.module_list:
            if isinstance(seq_module, torch.nn.Sequential):
                first_layer = seq_module[0]
                if isinstance(first_layer, Conv2dNormActivation):
                    conv_layer = first_layer[0]
                    if isinstance(conv_layer, torch.nn.Conv2d): head_in_channels.append(conv_layer.in_channels)
                    else: raise TypeError(f"Expected Conv2d, got {type(conv_layer)}")
                elif isinstance(first_layer, torch.nn.Conv2d): head_in_channels.append(first_layer.in_channels)
                else: raise TypeError(f"Unexpected layer type: {type(first_layer)}")
            else: raise TypeError(f"Expected Sequential, got {type(seq_module)}")

        if len(head_in_channels) != len(num_anchors):
            print(f"Warning: Length mismatch! Channels: {len(head_in_channels)}, Anchors: {len(num_anchors)}.")
        print(f"Successfully extracted input channels by inspection: {head_in_channels}")

    except Exception as e:
        print(f"ERROR: Failed to extract input channels: {e}")
        print("----- Model Head Structure -----"); print(model.head); print("-----------------------------")
        raise

    norm_layer = nn.BatchNorm2d 
    print(f"Using norm_layer: {norm_layer.__name__}")

    print(f"Creating new SSDLiteHead with - In Channels: {head_in_channels}, Num Anchors: {num_anchors}, Num Classes: {NUM_CLASSES}")
    try:
        new_head = SSDLiteHead(
            in_channels=head_in_channels,
            num_anchors=num_anchors,
            num_classes=NUM_CLASSES,
            norm_layer=norm_layer
        )
    except Exception as e:
         print(f"ERROR: Failed to instantiate new SSDLiteHead: {e}")
         print("Check if all required arguments are provided and parameters match.")
         raise

    model.head = new_head
    print(f"Model head successfully replaced for {NUM_CLASSES} classes.")

    return model

voc_cls_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}

def parse_voc_annotation(xml_file):
    if not os.path.exists(xml_file): warnings.warn(f"Anno file not found: {xml_file}"); return None, None, None
    try: tree = ET.parse(xml_file)
    except ET.ParseError: warnings.warn(f"Failed to parse XML: {xml_file}"); return None, None, None
    root = tree.getroot(); boxes = []; labels = []; img_size_info = root.find('size')
    orig_w_text = img_size_info.find('width').text if img_size_info and img_size_info.find('width') is not None else None
    orig_h_text = img_size_info.find('height').text if img_size_info and img_size_info.find('height') is not None else None
    try:
        orig_w = int(orig_w_text) if orig_w_text else None
        orig_h = int(orig_h_text) if orig_h_text else None
    except (ValueError, TypeError):
        warnings.warn(f"Could not parse original image dimensions from {xml_file}. Found w='{orig_w_text}', h='{orig_h_text}'.")
        orig_w, orig_h = None, None
    original_size_hw = (orig_h, orig_w) if orig_h is not None and orig_w is not None else None
    for obj in root.findall('object'):
        name_elem = obj.find('name'); bndbox = obj.find('bndbox');
        if name_elem is None or not name_elem.text or bndbox is None: continue;
        name = name_elem.text
        if name not in voc_cls_to_idx: continue;
        label = voc_cls_to_idx[name];
        if label == 0: continue;
        xmin_elem=bndbox.find('xmin');ymin_elem=bndbox.find('ymin');xmax_elem=bndbox.find('xmax');ymax_elem=bndbox.find('ymax')
        if None in [xmin_elem,ymin_elem,xmax_elem,ymax_elem] or any(c is None or not c.text for c in [xmin_elem,ymin_elem,xmax_elem,ymax_elem]): continue;
        try: xmin=float(xmin_elem.text);ymin=float(ymin_elem.text);xmax=float(xmax_elem.text);ymax=float(ymax_elem.text)
        except ValueError: continue;
        if orig_w is not None: xmin=max(0.,xmin);xmax=min(float(orig_w),xmax)
        if orig_h is not None: ymin=max(0.,ymin);ymax=min(float(orig_h),ymax)
        if xmin >= xmax or ymin >= ymax: continue;
        boxes.append([xmin, ymin, xmax, ymax]); labels.append(label)
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64), original_size_hw
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64), original_size_hw

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download=False, augmentations=None):
        self.root_dir = root
        try:
             super().__init__(root=root, year=year, image_set=image_set, download=download)
             self._weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
             self._model_preprocess = self._weights.transforms()
             cropper = getattr(self._model_preprocess, 'crop_size', None)
             resizer = getattr(self._model_preprocess, 'resize_size', None)
             if isinstance(cropper, (list, tuple)) and len(cropper) == 2:
                  self._target_size = tuple(cropper)
             elif isinstance(resizer, (list, tuple)) and len(resizer) >= 1:
                  if len(resizer) == 1: self._target_size = (resizer[0], resizer[0])
                  elif len(resizer) == 2: self._target_size = tuple(resizer)
                  else: self._target_size = (320, 320)
             else: self._target_size = (320, 320)
        except Exception as e_init:
             print(f"FATAL ERROR in VOCDetection __init__ or getting transforms: {e_init}")
             traceback.print_exc(); sys.exit(1)
        self.annotations_dir = os.path.join(root, 'VOCdevkit', f'VOC{year}', 'Annotations')
        if not os.path.isdir(self.annotations_dir): raise FileNotFoundError(f"Annotations dir missing: {self.annotations_dir}")
        self._augmentations = augmentations

    def __getitem__(self, index):
        img_pil = None; target_dict = None; annotation_path = None; original_size_hw = None
        try:
            img_pil, target_dict = super().__getitem__(index)
            if not isinstance(img_pil, Image.Image):
                try: img_pil = F.to_pil_image(img_pil)
                except Exception: pass
            if not isinstance(img_pil, Image.Image):
                 print(f"ERROR: Could not load index {index} as PIL Image. Got type: {type(img_pil)}."); return None
            original_w, original_h = img_pil.size
            original_size_hw = (original_h, original_w)
            if isinstance(target_dict, dict) and 'annotation' in target_dict:
                 anno_info = target_dict['annotation']
                 if 'filename' in anno_info and isinstance(anno_info['filename'], str):
                     filename = anno_info['filename']
                     if not filename: raise ValueError(f"Missing filename in annotation dict idx {index}.")
                     xml_filename = filename.replace('.jpg', '.xml').replace('.jpeg', '.xml')
                     annotation_path = os.path.join(self.annotations_dir, xml_filename)
                 else: raise ValueError(f"Cannot determine annotation path from target_dict for index {index}.")
            else: raise ValueError(f"Target dictionary format incorrect or missing 'annotation' key for index {index}.")
            if not annotation_path or not os.path.exists(annotation_path):
                 warnings.warn(f"Annotation path invalid or DNE: {annotation_path}. Skip idx {index}."); return None
        except IndexError: print(f"ERROR: Index {index} out of bounds."); return None
        except Exception as e_load: print(f"ERROR loading base data/finding path idx {index}: {e_load}"); traceback.print_exc(); return None
        try:
             boxes, labels, _ = parse_voc_annotation(annotation_path)
             if boxes is None: warnings.warn(f"Annotation parsing returned None for {annotation_path}. Skipping idx {index}."); return None
        except Exception as e_parse: print(f"ERROR parsing annotation {annotation_path} idx {index}: {e_parse}"); traceback.print_exc(); return None
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]),
                  'iscrowd': torch.zeros((boxes.shape[0],), dtype=torch.int64),
                  'original_size': torch.tensor(original_size_hw, dtype=torch.int64)}
        if boxes.shape[0] > 0: target['area'] = torch.clamp((boxes[:, 3]-boxes[:, 1])*(boxes[:, 2]-boxes[:, 0]), min=1.0)
        else: target['area'] = torch.tensor([], dtype=torch.float32)
        img_final = None; target_final = target.copy()
        try:
            if self._augmentations is not None: warnings.warn("Augmentation logic skipped. Ensure augmentations handle PIL & boxes if used.")
            target_h, target_w = self._target_size
            img_resized_pil = img_pil.resize((target_w, target_h), Image.BILINEAR)
            if target_final['boxes'].shape[0] > 0:
                 boxes_original_coords = target_final['boxes']
                 orig_h, orig_w = original_size_hw
                 h_ratio = target_h / orig_h if orig_h > 0 else 1.0
                 w_ratio = target_w / orig_w if orig_w > 0 else 1.0
                 boxes_scaled_to_target = boxes_original_coords.clone()
                 boxes_scaled_to_target[:, [0, 2]] *= w_ratio
                 boxes_scaled_to_target[:, [1, 3]] *= h_ratio
                 boxes_scaled_to_target[:, 0] = torch.clamp(boxes_scaled_to_target[:, 0], min=0, max=target_w)
                 boxes_scaled_to_target[:, 1] = torch.clamp(boxes_scaled_to_target[:, 1], min=0, max=target_h)
                 boxes_scaled_to_target[:, 2] = torch.clamp(boxes_scaled_to_target[:, 2], min=0, max=target_w)
                 boxes_scaled_to_target[:, 3] = torch.clamp(boxes_scaled_to_target[:, 3], min=0, max=target_h)
                 target_final['boxes'] = boxes_scaled_to_target
                 target_final['area'] = torch.clamp((boxes_scaled_to_target[:, 3]-boxes_scaled_to_target[:, 1])*(boxes_scaled_to_target[:, 2]-boxes_scaled_to_target[:, 0]), min=1.0)
            img_final = self._model_preprocess(img_resized_pil)
        except Exception as e_transform: print(f"ERROR during transformation/preprocessing idx {index}: {e_transform}"); traceback.print_exc(); return None
        if not isinstance(img_final, torch.Tensor): print(f"Warn: Final image is not a Tensor idx {index}. Type: {type(img_final)}"); return None
        if not isinstance(target_final, dict) or 'boxes' not in target_final or 'labels' not in target_final or 'original_size' not in target_final:
             print(f"Warn: Final target format incorrect idx {index}. Keys: {target_final.keys()}"); return None
        if not isinstance(target_final['boxes'], torch.Tensor) or \
           not isinstance(target_final['labels'], torch.Tensor) or \
           not isinstance(target_final['original_size'], torch.Tensor):
             print(f"Warn: Final target boxes/labels/original_size are not Tensors idx {index}"); return None
        target_final['boxes'] = target_final['boxes'].to(dtype=torch.float32)
        target_final['labels'] = target_final['labels'].to(dtype=torch.int64)
        target_final['image_id'] = target_final['image_id'].to(dtype=torch.int64)
        target_final['iscrowd'] = target_final['iscrowd'].to(dtype=torch.int64)
        target_final['area'] = target_final['area'].to(dtype=torch.float32)
        target_final['original_size'] = target_final['original_size'].to(dtype=torch.int64)
        return img_final, target_final

def collate_fn(batch):
    original_size = len(batch); batch_filtered = []
    for i, item in enumerate(batch):
         if item is None: continue
         if not isinstance(item, tuple) or len(item) != 2: continue
         img, tgt = item
         if not isinstance(img, torch.Tensor) or not isinstance(tgt, dict) or 'original_size' not in tgt:
              warnings.warn(f"Collate fn skipping item {i}: Invalid types or missing 'original_size'. img={type(img)}, tgt={type(tgt)}")
              continue
         batch_filtered.append(item)
    filtered_size = len(batch_filtered)
    if original_size != filtered_size: warnings.warn(f"Collate fn removed {original_size - filtered_size} None/invalid items.")
    if not batch_filtered: warnings.warn("Collate fn returning empty lists after filtering."); return ([], [])
    try: images, targets = list(zip(*batch_filtered)); return list(images), list(targets)
    except Exception as e: warnings.warn(f"Error collate zip: {e}. Return empty."); return ([], [])

def get_dataloaders(batch_size_local, num_workers_local):
    dataset_train = VOCDataset(root=VOC_ROOT_DIR, year=YEAR, image_set='trainval', download=False)
    dataset_val = VOCDataset(root=VOC_ROOT_DIR, year=YEAR, image_set='test', download=False)
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size_local, sampler=sampler_train,
        num_workers=num_workers_local, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, sampler=sampler_val,
        num_workers=num_workers_local, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    return loader_train, loader_val

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    print("\nRunning simplified validation (checking inference)...")
    print("Simplified validation check finished.")
    return {}

class SimpleMetricLogger:
    def __init__(self, delimiter="\t"): self.meters = {}; self.delimiter = delimiter
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor): v = v.item()
            if not math.isfinite(v): continue
            meter = self.meters.get(k);
            if meter is None: meter = {'total': 0.0, 'count': 0, 'avg': 0.0}; self.meters[k] = meter
            meter['total'] += v; meter['count'] += 1; meter['avg'] = meter['total'] / meter['count']
    def __str__(self):
        loss_str = []
        if 'loss' in self.meters: loss_str.append(f"loss: {self.meters['loss']['avg']:.4f}")
        for name, meter in self.meters.items():
             if name != 'loss':
                 if name in ['classification', 'bbox_regression']: loss_str.append(f"{name}: {meter['avg']:.4f}")
                 elif name == 'lr': loss_str.append(f"lr: {meter['avg']:.6f}")
        return self.delimiter.join(loss_str)
    def log_every(self, iterable, print_freq, header=None):
        i = 0; start_time = time.time(); end = time.time(); header = header or ''
        iter_time = {'total': 0.0, 'count': 0, 'avg': 0.0}; data_time = {'total': 0.0, 'count': 0, 'avg': 0.0}
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg_template = self.delimiter.join([ header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time:.4f}', 'data: {data:.4f}'])
        for obj in iterable:
            data_t = time.time() - end; data_time['total'] += data_t; data_time['count'] += 1; data_time['avg'] = data_time['total'] / data_time['count']
            yield obj
            iter_t = time.time() - end; iter_time['total'] += iter_t; iter_time['count'] += 1; iter_time['avg'] = iter_time['total'] / iter_time['count']
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time['avg'] * (len(iterable) - 1 - i); eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                meters_string = str(self)
                log_values = {"meters": meters_string, "time": iter_time['avg'], "data": data_time['avg'], "eta": eta_string}
                print(log_msg_template.format(i, len(iterable), **log_values))
            i += 1; end = time.time()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, use_warmup=False):
    model.train()
    metric_logger = SimpleMetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    lr_scheduler_warmup = None
    if use_warmup and epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        print(f"Applying linear warmup for {warmup_iters} iterations.")
        try:
            if hasattr(torch.optim.lr_scheduler, 'LinearLR'):
                 lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
            else:
                 print("Warning: torch.optim.lr_scheduler.LinearLR not found. Skipping warmup.")
        except Exception as e_warmup:
            print(f"Warning: Failed to initialize LinearLR scheduler: {e_warmup}. Skipping warmup.")
    num_iterations = len(data_loader)
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"ERROR: Loss is {loss_value}, stopping training.")
            print(loss_dict); sys.exit(1)
        optimizer.zero_grad(); losses.backward(); optimizer.step()
        if lr_scheduler_warmup is not None and epoch == 0 and i < warmup_iters:
            lr_scheduler_warmup.step()
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger

def main():
    print(f"Using device: {DEVICE}")
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    try:
        loader_train, loader_val = get_dataloaders(BATCH_SIZE, NUM_WORKERS)
    except ValueError as e: print(f"Error creating dataloaders: {e}"); return
    except Exception as e_gen: print(f"General error creating dataloaders: {e_gen}"); traceback.print_exc(); return
    try:
        model = get_object_detection_model()
        model.to(DEVICE)
    except Exception as e_mod: print(f"Error loading model: {e_mod}"); traceback.print_exc(); return
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    milestones = [100, 150]
    lr_scheduler_epoch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    print(f"Using MultiStepLR scheduler. Milestones: {milestones}, Gamma: 0.1")
    start_epoch = 0; apply_warmup = True
    if os.path.exists(LATEST_CHECKPOINT_FILE):
        print(f"Attempting to resume training from checkpoint: {LATEST_CHECKPOINT_FILE}")
        try:
            checkpoint = torch.load(LATEST_CHECKPOINT_FILE, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler_epoch.__class__.__name__ == checkpoint.get('scheduler_type', ''):
                lr_scheduler_epoch.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print("Warning: Checkpoint scheduler type mismatch. Using new scheduler state.")
                resumed_epoch = checkpoint['epoch']
                print(f"Manually setting scheduler state based on resumed epoch {resumed_epoch}...")
                if hasattr(lr_scheduler_epoch, 'last_epoch'):
                    lr_scheduler_epoch.last_epoch = resumed_epoch
                    print(f"Set scheduler's last_epoch to {lr_scheduler_epoch.last_epoch}")
                else:
                    for _ in range(resumed_epoch + 1): lr_scheduler_epoch.step()
                    print(f"Manually stepped scheduler {resumed_epoch+1} times.")
            start_epoch = checkpoint['epoch'] + 1
            apply_warmup = False
            print(f"Resuming from epoch {start_epoch}.")
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint correctly: {e}")
            print("Consider deleting checkpoints and starting fresh."); traceback.print_exc()
            start_epoch = 0; apply_warmup = True
    else:
        print("No checkpoint found. Starting training from scratch."); apply_warmup = True
    print(f"\nStarting training loop from epoch {start_epoch} up to {NUM_EPOCHS-1}...")
    start_time = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS):
        use_warmup_this_epoch = (epoch == 0 and apply_warmup)
        epoch_logger = train_one_epoch(model, optimizer, loader_train, DEVICE, epoch, PRINT_FREQ, use_warmup_this_epoch)
        lr_scheduler_epoch.step()
        avg_loss = epoch_logger.meters.get('loss', {}).get('avg', float('nan'))
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print(f"End of Epoch Learning Rate (for next epoch): {current_lr:.6f}")
        is_last_epoch = (epoch == NUM_EPOCHS - 1)
        if (epoch + 1) % CHECKPOINT_SAVE_FREQ == 0 or is_last_epoch:
            checkpoint_data = {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler_epoch.state_dict(),
                'scheduler_type': lr_scheduler_epoch.__class__.__name__,
                'config_batch_size': BATCH_SIZE,
            }
            epoch_filename = f"{EPOCH_CHECKPOINT_PREFIX}{epoch}.pth"
            print(f"\nSaving epoch checkpoint to {epoch_filename}...")
            torch.save(checkpoint_data, epoch_filename)
            print(f"Updating latest checkpoint file: {LATEST_CHECKPOINT_FILE}...")
            torch.save(checkpoint_data, LATEST_CHECKPOINT_FILE)
        print("-" * 25 + "\n")
    print("Training finished.")
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Final model state_dict saved to {MODEL_SAVE_PATH}")
    except Exception as e: print(f"ERROR: Failed to save final model state_dict: {e}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total Training time: {total_time_str}')

if __name__ == "__main__":
    if not os.path.isdir(VOC_ROOT_DIR):
        print(f"ERROR: VOC_ROOT_DIR path not found: '{VOC_ROOT_DIR}'")
        print("Please ensure the dataset path is correct."); sys.exit(1)
    main()