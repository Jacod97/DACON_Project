from pathlib import Path
import os
import random
import numpy as np
import torch
import shutil

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def copy_image(image, split_dir):
    for img_path in image:
        image_name = Path(img_path).name
        copy_path = split_dir / image_name
        shutil.copy(img_path, copy_path)