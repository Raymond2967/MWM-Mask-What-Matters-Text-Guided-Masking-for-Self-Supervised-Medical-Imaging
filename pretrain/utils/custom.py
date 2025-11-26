import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
from tqdm import tqdm

def calculate_dataset_stats(dataset_root, input_size, max_samples=None):
    print("正在计算数据集统计信息...")
    
    images_dir = os.path.join(dataset_root, 'images')
    img_names = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    if max_samples is not None:
        img_names = img_names[:max_samples]
    
    print(f"使用 {len(img_names)} 张图片计算统计信息")
    
    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)
    pixel_count = 0
    
    transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  
        ToTensorV2()
    ])
    
    for img_name in tqdm(img_names, desc="计算统计信息"):
        img_path = os.path.join(images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用transform
        transformed = transform(image=image)
        image_tensor = transformed['image']  # shape: [C, H, W]
        
        for i in range(3):
            channel_data = image_tensor[i].numpy().flatten()
            mean_sum[i] += np.mean(channel_data)
            std_sum[i] += np.std(channel_data)
        
        pixel_count += 1
    
    mean = mean_sum / pixel_count
    std = std_sum / pixel_count
    
    print(f"计算完成！")
    print(f"均值 (R, G, B): {mean}")
    print(f"标准差 (R, G, B): {std}")
    
    return mean.tolist(), std.tolist()

class MyCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_expanded_dir = os.path.join(root_dir, 'masks_expanded')
        self.img_names = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_expanded_path = os.path.join(self.masks_expanded_dir, img_name)
        # 读取图片和mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_expanded = cv2.imread(mask_expanded_path, 0)  # 单通道
        # 动态resize mask到和image一致（为后续扩展保留）
        if mask_expanded.shape[:2] != image.shape[:2]:
            mask_expanded = cv2.resize(mask_expanded, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # albumentations transform
        if self.transform:
            transformed = self.transform(image=image, mask=mask_expanded)
            image = transformed['image']
            mask_expanded = transformed['mask']  # 现在返回mask
        return image, mask_expanded  # 返回图像和mask的元组

def get_train_transform(input_size, mean=None, std=None):
    if mean is None:
        mean = IMAGENET_DEFAULT_MEAN
    if std is None:
        std = IMAGENET_DEFAULT_STD
    
    return A.Compose([
        A.Resize(input_size, input_size),  
        A.HorizontalFlip(),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

def build_dataset_to_pretrain(dataset_root, input_size, use_custom_stats=True, max_samples_for_stats=1000):

    mean, std = None, None
    
    if use_custom_stats:
        try:
            # 尝试计算自定义统计信息
            mean, std = calculate_dataset_stats(dataset_root, input_size, max_samples_for_stats)
            print(f"使用自定义统计信息: mean={mean}, std={std}")
        except Exception as e:
            print(f"计算自定义统计信息失败: {e}")
            print("使用ImageNet默认统计信息")
            mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        print(f"使用ImageNet默认统计信息: mean={mean}, std={std}")
    
    trans_train = get_train_transform(input_size, mean, std)
    dataset_train = MyCustomDataset(root_dir=dataset_root, transform=trans_train)
    print('Using albumentations transform for image & mask_expanded (同步变换，无随机裁剪)')
    return dataset_train

def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')

