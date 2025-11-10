import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from modules.msdaunet import MSDAUNet
from configs.config_setting import setting_config
from utils import get_logger, log_config_info, save_imgs, myNormalize, myToTensor, myResize
import os
from tqdm import tqdm

# 加载配置
config = setting_config()

# 实例化模型
model = MSDAUNet(num_classes=config.num_classes, input_channels=config.input_channels, c_list=config.model_config['c_list'], bridge=config.model_config['bridge'], gt_ds=config.model_config['gt_ds'])

# 加载权重
try:
    model.load_state_dict(torch.load(config.pretrained_path + 'best-epoch119-loss0.7047.pth'))
    model.eval()  # 设置为评估模式
except Exception as e:
    print(f"Failed to load model weights: {e}")
    exit()

def test_one_image(img_path, msk_path, model, config, save_path, threshold=0.5):
    # 加载图片并预处理
    img = Image.open(img_path)
    msk = Image.open(msk_path)

    # 应用测试变换
    normalize = myNormalize(config.datasets, train=False)
    to_tensor = myToTensor()
    resize = myResize(config.input_size_h, config.input_size_w)

    img, msk = normalize((np.array(img), np.array(msk)))
    img, msk = to_tensor((img, msk))
    img, msk = resize((img, msk))

    batch_t = torch.unsqueeze(img, 0)  # 增加一个批次维度

    # 进行预测
    with torch.no_grad():  # 关闭梯度计算
        output = model(batch_t)

    # 打印输出结构
    print(f"Model output structure: {output}")

    # 假设最后一个输出是你需要的预测掩码
    if isinstance(output, tuple):
        pred_mask_tensor = output[-1]  # 提取最后一个张量作为预测掩码
    else:
        pred_mask_tensor = output

    # 确保pred_mask_tensor是一个张量
    if not isinstance(pred_mask_tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor, but got {type(pred_mask_tensor)}")

    # 后处理
    pred_mask = pred_mask_tensor.squeeze().cpu().numpy()  # 去掉批次维度并转换为numpy数组
    pred_mask = (pred_mask > threshold).astype(np.uint8)  # 二值化

    # 确保pred_mask的形状是 (1, H, W)
    if len(pred_mask.shape) == 2:
        pred_mask = np.expand_dims(pred_mask, axis=0)  # 添加一个批次维度

    # 保存预测图像
    os.makedirs(save_path, exist_ok=True)  # 确保目录存在
    save_imgs(img, msk, pred_mask, 0, save_path, config.datasets, threshold=threshold)

    # 返回预测结果
    return img, msk, pred_mask

# 测试图像和掩码的路径列表
test_images = [
    # ('./testpng/images/336.png', './testpng/masks/336.png'),
    # ('./testpng/images/439.png', './testpng/masks/439.png'),
    # ('./testpng/images/498.png', './testpng/masks/498.png'),
    ('./testpng/images/709.png', './testpng/masks/709.png'),
    # 添加更多图像路径
]

# 保存预测结果的路径
save_path = 'outputs/'

# 收集所有预测结果
all_results = []

# 测试每张图像
for img_path, msk_path in test_images:
    img, msk, pred_mask = test_one_image(img_path, msk_path, model, config, save_path, threshold=config.threshold)
    all_results.append((img, msk, pred_mask))

# 显示所有预测结果
plt.figure(figsize=(15, 5 * len(all_results)))

for i, (img, msk, pred_mask) in enumerate(all_results):
    plt.subplot(len(all_results), 3, 3 * i + 1)
    plt.imshow(img.permute(1, 2, 0).numpy() / 255.0)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(len(all_results), 3, 3 * i + 2)
    plt.imshow(msk.squeeze(0).numpy(), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(len(all_results), 3, 3 * i + 3)
    plt.imshow(pred_mask[0], cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

plt.tight_layout()
plt.show()
