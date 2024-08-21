import os
import torch
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture
from PIL import Image
import torchvision.transforms as transforms

class InferenceDataset(Dataset):
    def __init__(self, config: Config, img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms

        # 현재 경로에 존재하는 파일들만 리스트업
        self.video_files = [f for f in os.listdir(self.videos_dir) if os.path.isfile(os.path.join(self.videos_dir, f))]
        print(f"Found {len(self.video_files)} video files for inference.")

    def __getitem__(self, index):
        video_file = self.video_files[index]
        video_path = os.path.join(self.videos_dir, video_file)
        
        # 비디오 프레임 로드
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)

        if self.img_transforms is not None:
            # Tensor를 PIL 이미지로 변환 후, img_transforms 적용
            imgs = [self.img_transforms(transforms.ToPILImage()(img)) for img in imgs]

        return {
            'video_id': video_file,
            'video': torch.stack(imgs),  # 최종적으로 tensor 리스트를 스택하여 배치 형태로 변환
        }

    def __len__(self):
        return len(self.video_files)
