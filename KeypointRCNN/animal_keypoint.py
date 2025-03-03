import os
import cv2
import numpy as np
import pandas as ps
from typing import Tuple, Sequence, Callable, Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset

class KeypointDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        label_df: pd.DataFrame,
        transforms: Sequence[Callable]=None
    ) -> None:
        self.image_dir = image_dir
        self.df = label_df
        self.transforms = transforms

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        image_id = self.df.iloc[index, 0]
        labels = np.array([1])
        keypoints = self.df.iloc[index, 1:].values.reshape(-1, 2).astype(np.int64)

        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        
        # 키포인트 중 0이 아닌 값만 필터링 (유효한 키포인트만 고려)
        valid_keypoints = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]

        # 유효한 키포인트가 있는 경우에만 바운딩 박스 생성
        if len(valid_keypoints) > 0:
            x1, y1 = np.min(valid_keypoints[:, 0]), np.min(valid_keypoints[:, 1])
            x2, y2 = np.max(valid_keypoints[:, 0]), np.max(valid_keypoints[:, 1])
            # 바운딩 박스 크기 보정 (너비와 높이가 최소한 1 이상이 되도록)
            if x1 == x2:
                x2 += 1
            if y1 == y2:
                y2 += 1
            boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)
        else:
            # 유효한 키포인트가 없으면 바운딩 박스를 [0, 0, 1, 1]로 설정 (임의의 작은 값)
            boxes = np.array([[0, 0, 1, 1]], dtype=np.int64)

        image = cv2.imread(os.path.join(self.image_dir, image_id), cv2.COLOR_BGR2RGB)

        targets ={
            'image': image,
            'bboxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        if self.transforms is not None:
            targets = self.transforms(**targets)

        image = targets['image']
        image = image / 255.0

        targets = {
            'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
            'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
            'keypoints': torch.as_tensor(
                np.concatenate([targets['keypoints'], np.ones((15, 1))], axis=1)[np.newaxis], dtype=torch.float32
            )
        }

        return image, targets
