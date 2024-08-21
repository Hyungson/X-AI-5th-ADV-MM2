import os
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from config.base_config import Config
from datasets.video_capture import VideoCapture

class CustomDataset(Dataset):

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        pth = 'data/hollywood2/'
        
        if split_type == 'train':
            self.label_csv = os.path.join(pth, 'train.csv')
            train_df = pd.read_csv(self.label_csv)
            self.train_vids = train_df['video_name'].unique()
            self.train_df = train_df  # train_df를 나중에 사용할 수 있도록 저장
            self._compute_vid2caption()
            self._construct_all_train_pairs()

        elif split_type == 'test':
            self.label_csv = os.path.join(pth, 'test.csv')
            self.test_df = pd.read_csv(self.label_csv)
        else:
            raise NotImplementedError('Unseen data split type!')

    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id, sen_id = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                             self.config.num_frames,
                                                             self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = [self.img_transforms(img) for img in imgs]  # 각 프레임에 대해 변환 적용

            return {
                'video_id': video_id,
                'video': torch.stack(imgs),  # 변환된 프레임을 쌓아 텐서로 변환
                'text': caption,
            }
        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                             self.config.num_frames,
                                                             self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = [self.img_transforms(img) for img in imgs]  # 각 프레임에 대해 변환 적용

            return {
                'video_id': video_id,
                'video': torch.stack(imgs),  # 변환된 프레임을 쌓아 텐서로 변환
                'text': caption,
            }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)

    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption, senid = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.avi')
            return video_path, caption, vid, senid
        else:
            vid = self.test_df.iloc[index].video_name
            caption = self.test_df.iloc[index].sentence
            video_path = os.path.join(self.videos_dir, vid + '.avi')
            return video_path, caption, vid

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption, senid in zip(self.vid2caption[vid], self.vid2senid[vid]):
                    self.all_train_pairs.append([vid, caption, senid])

    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        self.vid2senid = defaultdict(list)

        for _, row in self.train_df.iterrows():
            vid = row['video_name']
            caption = row['sentence']
            senid = row.name  # index 값을 이용하여 senid 설정
            self.vid2caption[vid].append(caption)
            self.vid2senid[vid].append(senid)
