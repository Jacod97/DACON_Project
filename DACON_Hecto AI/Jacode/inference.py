import torch
import torch.nn.functional as F
import pandas as pd
from config import device
from model import BaseModel
from dataset import get_dataloaders

def inference(test_loader, class_names):
    # 저장된 모델 로드
    model = BaseModel(num_classes=len(class_names))
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)

    # 추론
    model.eval()
    results = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            # 각 배치의 확률을 리스트로 변환
            for prob in probs.cpu():
                result = {
                    class_names[i]: prob[i].item()
                    for i in range(len(class_names))
                }
                results.append(result)

    return pd.DataFrame(results)

if __name__ == "__main__":
    train_root = './train'
    test_root = './test'
    
    # 데이터로더와 클래스 이름 가져오기
    _, _, test_loader, class_names = get_dataloaders(train_root, test_root)
    
    # 추론 실행
    pred = inference(test_loader, class_names)
    
    # 제출 파일 생성
    submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
    class_columns = submission.columns[1:]
    pred = pred[class_columns]
    submission[class_columns] = pred.values
    submission.to_csv('baseline_submission.csv', index=False, encoding='utf-8-sig') 