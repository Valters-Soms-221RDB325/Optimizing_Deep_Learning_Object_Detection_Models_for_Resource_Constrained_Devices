import yaml

voc_2007_yaml = {
    'path': './datasets/VOC',
    'train': ['images/train2007', 'images/val2007'],
    'val': ['images/test2007'],
    'test': ['images/test2007'],
    'names': {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }
}

with open('VOC2007.yaml', 'w') as f:
    yaml.dump(voc_2007_yaml, f, sort_keys=False)
    
print("Created VOC2007.yaml file that only includes the 2007 dataset") 