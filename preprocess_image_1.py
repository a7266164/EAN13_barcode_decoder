import numpy as np
import torch

# 優化完畢

# 1. 彩圖轉灰階
def custom_cvtColor(image):
  
     # 檢查圖像通道數
    if len(image.shape) == 2:
        # print("輸入圖像已經是灰度圖像，無需轉換")
        return image

    # 使用 NumPy 索引來獲取每個通道
    blue = image[:,:,0]
    green = image[:,:,1]
    red = image[:,:,2]

    # 根據公式進行轉換
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gray = gray.astype(np.uint8)
    
    return gray

# 2. 調整圖片大小
def bilinear_resize(image, new_width, new_height):
    
    # 轉換為 PyTorch 張量並添加批次和通道維度
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    # 使用 PyTorch 的 interpolate 函數進行縮放
    output_tensor = torch.nn.functional.interpolate(image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=True)

    # 轉換回 NumPy 陣列
    output_image = output_tensor.squeeze(0).squeeze(0).numpy().astype(np.uint8)
    
    return output_image

# 3. 適應性二值化(中心區塊)
def adaptive_thresholding_center(image, block_size=200, C=18, n=10):

    # 確保 n 不大於 block_size
    n = min(n, block_size)
    
    # 設定填充量，使鄰近區域可以落在圖像中心
    pad_size = block_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    
    # 初始化輸出圖像
    binarized = np.zeros_like(image, dtype=np.uint8)
    
    # 遍歷每一個像素，步長為 n
    for i in range(0, image.shape[0], n):
        for j in range(0, image.shape[1], n):
            
            # 取出鄰近區域
            local_region = padded_image[i:i+block_size, j:j+block_size]
            
            # 計算平均值
            mean_value = np.mean(local_region)
            
            # 遍歷中心的 n x n 像素
            for x in range(n):
                for y in range(n):
                    if i+x < image.shape[0] and j+y < image.shape[1]:  # 確保不超出邊界
                        # 比較當前像素與平均值，決定二值化結果
                        if image[i+x, j+y] > mean_value - C:
                            binarized[i+x, j+y] = 255
                        else:
                            binarized[i+x, j+y] = 0
    
    return binarized

#========================================棄用函數========================================#

'''

# from scipy.signal import convolve2d

# # 4. 全域二值化
# def global_thresholding(image, C=100):
#     # 初始化輸出圖像
#     binarized = np.zeros_like(image, dtype=np.uint8)
    
#     # 計算平均值（全域平均）
#     mean_value = np.mean(image)
    
#     # 遍歷每一個像素
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             # 比較當前像素與全域平均值，決定二值化結果
#             if image[i, j] > mean_value - C:
#                 binarized[i, j] = 255
#             else:
#                 binarized[i, j] = 0
    
#     return binarized

# # 5. 適應性二值化
# def adaptive_thresholding(image, block_size=200, C=18):

#     start_time = time.time()

#     # 設定填充量，使鄰近區域可以落在圖像中心
#     pad_size = block_size // 2
#     padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    
#     # 初始化輸出圖像
#     binarized = np.zeros_like(image, dtype=np.uint8)
    
#     # 遍歷每一個像素
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             # 取出鄰近區域
#             local_region = padded_image[i:i+block_size, j:j+block_size]
            
#             # 計算平均值
#             mean_value = np.mean(local_region)
            
#             # 比較當前像素與平均值，決定二值化結果
#             if image[i, j] > mean_value - C:
#                 binarized[i, j] = 255
#             else:
#                 binarized[i, j] = 0
    
#     end_time = time.time()

#     cost = end_time - start_time
#     print(f"花費時間:{cost}")

#     return binarized

# # 6. 區塊性適應性二值化
# def block_based_adaptive_thresholding(image, block_size=200, C=18):
    
#     # 獲取圖像的尺寸
#     rows, cols = image.shape
    
#     # 初始化輸出圖像
#     binarized = np.zeros_like(image, dtype=np.uint8)
    
#     # 遍歷每個區塊
#     for i in range(0, rows, block_size):
#         for j in range(0, cols, block_size):
#             # 獲取當前區塊
#             block = image[i:i+block_size, j:j+block_size]
            
#             # 計算區塊的中位數
#             median_value = np.median(block)
            
#             # 進行二值化
#             binarized[i:i+block_size, j:j+block_size] = np.where(block > median_value - C, 255, 0)

#     return binarized

# # 7. 二值化加速版
# def adaptive_thresholding_fast(image, block_size=200, C=18):

#     kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size ** 2)
#     local_mean = convolve2d(image, kernel, 'same')
    
#     binarized = np.where(image > local_mean - C, 255, 0).astype(np.uint8)

#     return binarized

'''