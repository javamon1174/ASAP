import os
import time
import torch
import torch.optim as optim
from model import get_model
from data import load_data
import torch.nn.functional as F

def train(model, train_loader, optimizer, epoch, device = 'cuda'):
    model.train()                                        
    for batch_idx, (images, targets) in enumerate(train_loader):
        # data, target 값 DEVICE에 할당
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  

        optimizer.zero_grad()                 # optimizer gradient 값 초기화
        losses = model(images, targets)       # calculate loss

        loss = losses['loss_keypoint']        # keypoint loss
        loss.backward()                       # loss back propagation
        optimizer.step()                      # parameter update

        if (batch_idx+1) % 200 == 0:
            print(f'| epoch: {epoch} | batch: {batch_idx+1}/{len(train_loader)}')

def evaluate(model, test_loader, device = 'cuda'):
    model.train()      
    test_loss = 0      # test_loss 초기화
    
    with torch.no_grad(): 
        for images, targets in test_loader:
            # data, target 값 DEVICE에 할당
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  

            losses = model(images, targets)                       # validation loss
            test_loss += float(losses['loss_keypoint'])           # sum of all loss 
    
    test_loss /= len(test_loader.dataset)                         # 평균 loss
    return test_loss

def train_model(train_loader, val_loader, num_epochs = 30, device= 'cuda'):
    model_path = '/content/drive/MyDrive/models/RCNN.pt'
    
    # 기존 모델이 있으면 로드, 없으면 새 모델 생성
    if os.path.exists(model_path):
        print("✅ 기존 모델을 불러옵니다...")
        model = torch.load(model_path, map_location=device)
    else:
        print("🆕 새 모델을 생성합니다...")
        model = get_model()
    # model = get_model()
    
    model.to(device)
    
    best_loss = 999999  # initialize best loss
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, num_epochs+1):
        since = time.time()
        train(model, train_loader, optimizer, epoch, device)
        train_loss = evaluate(model, train_loader)
        val_loss = evaluate(model, val_loader)

        if val_loss <= best_loss:   # update best loss
          best_loss = val_loss
          torch.save(model, model_path)
          # torch.save(model, '../models/RCNN_ep'+str(epoch)+'_'+str(best_loss)+'.pt')
          print('Best Model Saved, Loss: ', val_loss)
        
        time_elapsed = time.time()-since
        print()
        print('---------------------- epoch {} ------------------------'.format(epoch))
        print('Train Keypoint Loss: {:.4f}, Val Keypoint Loss: {:.4f}'.format(train_loss, val_loss))   
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print()


def extract_confidence_scores(outputs):
    """ 모델의 출력에서 confidence score를 추출하는 함수 """
    confidence_scores = []
    
    for output in outputs:
        if 'scores' in output:
            confidence_scores.append(output['scores'].cpu().numpy())  # scores 리스트 저장
        else:
            confidence_scores.append([])  # scores가 없을 경우 빈 리스트 추가
    
    return confidence_scores


def evaluate_with_confidence(model, test_loader, device='cuda', output_json='confidence_scores.json'):
    """ 모델 평가 및 confidence score 추출 후 JSON으로 저장 """
    model.eval()  # 평가 모드
    confidence_results = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  

            outputs = model(images)  # 모델 예측 결과
            
            confidence_scores = extract_confidence_scores(outputs)  # 수정된 함수 사용
            for idx, score in enumerate(confidence_scores):
                confidence_results.append({
                    "labels": targets[idx]["labels"].item(),
                    "confidence_score": score.tolist()  # numpy 배열을 리스트로 변환
                })

    # JSON 파일로 저장
    with open(output_json, "w") as f:
        json.dump(confidence_results, f, indent=4)

    print(f"✅ Confidence scores saved to {output_json}")


def get_confidence():
    """ 기존 모델을 로드하고 confidence score를 계산하는 함수 """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 데이터 로드
    train_img_path = '../images/images_1'
    train_key_path = '/content/drive/MyDrive/annotations_1.csv'
    _, valid_loader = load_data(train_img_path, train_key_path)  # 검증 데이터 로드

    # 기존 모델 로드
    model_path = '/content/drive/MyDrive/models/RCNN.pt'
    if os.path.exists(model_path):
        print("✅ 기존 모델을 불러옵니다...")
        model = torch.load(model_path, map_location=DEVICE)
    else:
        print("🆕 새 모델을 생성합니다...")
        model = get_model()

    model.to(DEVICE)

    # Confidence score 추출 및 저장
    evaluate_with_confidence(model, valid_loader, device=DEVICE)


def main():
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_img_path = '../images/images_1'
    train_key_path = '/content/drive/MyDrive/annotations_1.csv'

    train_loader, valid_loader = load_data(train_img_path, train_key_path)
    train_model(train_loader, valid_loader, num_epochs = 10, device = DEVICE) 
    '''
    default: epoch - 30, 
             device - cuda
    '''
if __name__=="__main__":
    # get_confidence()  # 값 추출
    main()  # 훈련
