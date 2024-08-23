import os

import cv2
import numpy as np
import torch
import torch.utils.data
import json


# 读取词表并生成词汇表的索引映射
def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            word = line.strip()
            vocab[word] = idx
    return vocab

# 将文本转换为索引列表
def text_to_indices(text, vocab):
    indices = [vocab[word] for word in text.split() if word in vocab]
    return indices


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
                ├── HNS1_DATA
                    ├── images
                        ├── sparate
                                |   ├── 258_029_lung.png
                                |   ├── 258_032_heart.png
                                |   ├── ...

                        ├── whole
                                |   ├── 258_029.png
                                |   ├── 258_032.png
                                |   ├── ...

                    ├── mask
                        ├── 0
                            |   ├── 366_029.png
                            |   ├── 367_029.png
                            |   ├── ...

                        ├── 1
                            |   ├── 371_029.png
                            |   ├── 372_029.png
                                    ...
                .....
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        
        # 从 JSON 文件中加载字典
        with open('anatomy_dict.json', 'r') as json_file:
            self.loaded_dict = json.load(json_file)
        # 分割类别
        self.num_classes = num_classes
        self.transform = transform

        self.vocab = load_vocab('components/vocab.txt')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))  # 读图进

        # 获取先验特征
        prior_features = img_id[8:]
        Organ_num = self.loaded_dict[prior_features]
        text = 'The organs in the picture include ' + prior_features
        # 将文本转换为索引
        embedding = text_to_indices(text, self.vocab)
        embedding_array = np.array(embedding)
        # 假设变量定义
        mask_path = os.path.join(self.mask_dir, str(Organ_num), img_id[:7] + self.mask_ext)

        # 判断文件是否存在
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # 文件不存在，生成一个192x192的全0数组
            mask = np.zeros((192, 192), dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)  # 这个包比较方便，能把mask也一并做掉
            img = augmented['image']  # 参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']

        img = img.astype('float32') / 255  # 像素值缩放到0-1之间
        img = img.transpose(2, 0, 1)  # 将通道数放在前
        mask = mask.astype('float32') / 255

        return img, mask, embedding_array, Organ_num, {'img_id': img_id}
