import os
import glob
import shutil

root = "../data/train"

def move_dir(root, rm_dir, merge_dir):
    rm_path = os.path.join(root, rm_dir)
    merge_path = os.path.join(root, merge_dir)

    img_list = glob.glob(os.path.join(rm_path, "*.jpg"))
    print(f"{rm_dir}에 {len(img_list)}개 파일 확인.")

    for img in img_list:
        fname = os.path.basename(img)
        
        dst = os.path.join(merge_path, fname)
        shutil.move(img, dst)
        print(f"{fname} 이동 완료.")
        
if __name__ == "__main__":
    for fname in os.listdir(root):
        if fname == 'K5_3세대_하이브리드_2020_2022':
            move_dir(root, fname, "K5_하이브리드_3세대_2020_2023")
            shutil.rmtree(os.path.join(root, fname))
            print(f"{os.path.join(root, fname)}삭제 완료")

        elif fname == '디_올뉴니로_2022_2025':
            move_dir(root, fname, '디_올_뉴_니로_2022_2025')
            shutil.rmtree(os.path.join(root, fname))
            print(f"{os.path.join(root, fname)}삭제 완료")

        elif fname == '718_박스터_2017_2024':
            shutil.rmtree(os.path.join(root, fname))
            move_dir(root, fname, "박스터_718_2017_2024")
            print(f"{os.path.join(root, fname)}삭제 완료")