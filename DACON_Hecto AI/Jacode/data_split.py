from utils import copy_image
from pathlib import Path
import os
import random

train_root = r"C:\wanted\Git_project\DACON\DACON_Project\DACON_Hecto AI\data\train"
total_train_cnt = 0
total_val_cnt = 0

data_path = Path(train_root).parent
train_split_path = data_path / "train_split"
val_split_path = data_path / "val_split"

# if train_split_path.exists():
#     shutil.rmtree(train_split_path) # 폴더와 내용물 모두 삭제
# if val_split_path.exists():
#     shutil.rmtree(val_split_path) # 폴더와 내용물 모두 삭제

train_split_path.mkdir(parents=True, exist_ok=True)
val_split_path.mkdir(parents=True, exist_ok=True)

for fname in os.listdir(train_root):
    img_path = Path(train_root) / fname
    img_list = list(img_path.glob("*.jpg"))
    ### image count
    img_cnt = len(img_list)
    val_cnt = img_cnt*20//100
    train_cnt = img_cnt - val_cnt
    total_train_cnt += train_cnt
    total_val_cnt += val_cnt
    print(f"[{fname}]\n total : {img_cnt}개 train : {train_cnt}개 val : {val_cnt}개")
    ### train_vallidation_split
    random.shuffle(img_list)

    val_img = img_list[:val_cnt]
    train_img = img_list[val_cnt:]

    (train_split_path / fname).mkdir(parents=True, exist_ok=True)
    (val_split_path / fname).mkdir(parents=True, exist_ok=True)

    ### copy from train to split
    val_dir = val_split_path / fname
    train_dir = train_split_path / fname
    copy_image(val_img, val_dir)
    copy_image(train_img, train_dir)

print(f"\ntotal_train_count : {total_train_cnt} total val count {total_val_cnt}")
print("끝끝")