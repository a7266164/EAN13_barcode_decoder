import sys
import math
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
sys.path.append("C:/Users/a1234/Desktop/python/EAN13-barcode/2_locate_barcode/2_1_edge_detection")
import edge_detection_2_1

# 優化完畢

#=========================================主函數=========================================#

# 1. Canny邊緣檢測
def canny_edge_detection(image):

    # 高斯平滑
    # smoothed = edge_detection_2_1.gaussian_smoothing(image)

    # 梯度計算
    magnitude, angle = edge_detection_2_1.gradient_computation(image)

    # 非極大值抑制
    # suppression = edge_detection_2_1.non_maximum_suppression(magnitude, angle)

    # 雙閾值
    thresholding = edge_detection_2_1.double_thresholding(image = magnitude, low_threshold=50, high_threshold=150)

    # 邊緣追蹤
    edge_image = edge_detection_2_1.edge_tracking(image = thresholding, weak=100, strong=255)

    # 圖像正規化
    normalize = edge_detection_2_1.custom_normalize(edge_image, 0, 255, 'uint8')

    return normalize

# 2. Suzuki輪廓追蹤 
def suzuki_contour_tracing(binary_image):
    """
    Implement Suzuki's contour tracing algorithm on a given binary image.

    Parameters:
        binary_image (numpy.ndarray): The binary input image.

    Returns:
        list: List of contours, where each contour is a list of points (x, y).
    """
    # 初始化標記矩陣
    visited = np.zeros_like(binary_image, dtype=np.uint8)
    
    # 八個方向: 上, 右上, 右, 右下, 下, 左下, 左, 左上
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    
    # 儲存輪廓的列表
    all_contours = []
    
    rows, cols = binary_image.shape
    
    # 逐一確認二值化影像的每個像素
    for x in range(rows):
        for y in range(cols):
            if binary_image[x, y] == 255 and visited[x, y] == 0:
                # 找一個尚未被確認的像素
                
                # 初始化輪廓列表、起始點
                contour = []
                contour.append((x, y))
                
                # 標記看過的像素點
                visited[x, y] = 1
                
                # 初始化方向(向上)
                direction = 0
                
                # 輪廓追蹤
                while True:
                    found_next = False
                    
                    # 尋找輪廓的下一個點
                    for i in range(8):
                        # 計算新方向
                        new_direction = (direction + i) % 8
                        
                        # 計算候選像素點
                        new_x = x + dx[new_direction]
                        new_y = y + dy[new_direction]
                        
                        # 如果候選像素點確實是邊緣且尚未被標記
                        if 0 <= new_x < rows and 0 <= new_y < cols:
                            if binary_image[new_x, new_y] == 255 and visited[new_x, new_y] == 0:
                                # 尋找輪廓的下一個點
                                x, y = new_x, new_y
                                direction = (new_direction + 4) % 8  # 反轉方向
                                
                                # 將該點加入輪廓並標記
                                contour.append((x, y))
                                visited[x, y] = 1
                                
                                found_next = True
                                break
                    
                    if not found_next:
                        # 該輪廓蒐尋完畢，關閉迴圈
                        break
                
                # 將該輪廓添加到總輪廓列表
                all_contours.append(contour)
    
    return all_contours

# 3. 計算contour面積
def contour_area(contour):
    n = len(contour)
    if n < 3:  # A contour with fewer than 3 points can't enclose an area
        return 0.0
    
    area = 0.0
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    
    area = abs(area) / 2.0
    return area

# 4. 過濾contours
def filter_contours(contours):
    """
    過濾輪廓和相應的梯度方向。

    參數:
        contours (list): 輪廓列表。
        image (numpy.ndarray): 灰度圖像。
        gradient_direction (numpy.ndarray): 梯度方向圖。

    返回:
        filtered_contours (list): 過濾後的輪廓列表。
        filtered_gradient_direction (numpy.ndarray): 過濾後的梯度方向圖。
    """
    filtered_contours = []

    for contour in contours:
        # 矩形面積
        area = contour_area(contour)
        if area < 200 or area > 5000:
            continue

        # 計算輪廓的周長
        perimeter = contour_perimeter(contour)

        # # 計算邊界矩陣
        # x, y, w, h = boundingRect(contour)

        # # 黑白區域的交替（image 為灰度圖像）
        # line_profile = np.sum(image[y:y+h, x:x+w], axis=0)
        # peaks = (line_profile[1:] - line_profile[:-1]) > 0
        # num_transitions = np.sum(peaks[1:] != peaks[:-1])

        # if num_transitions > 10:
        #     continue

        # 矩形周長
        if perimeter > 200:
            filtered_contours.append(contour)
            
    return filtered_contours

# 5. 合併鄰近矩形
def merge_rectangles(rectangles, distance_threshold=80):

     # Calculate the center of each rectangle and store it in a list
    centers = np.array([[x + w / 2, y + h / 2] for x, y, w, h in rectangles])
    
    # Use DBSCAN to cluster the rectangle centers
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(centers)
    
    # Initialize list to store the merged rectangles
    merged_rectangles = []
    
    # Loop through each cluster and merge the rectangles in that cluster
    for label in set(clustering.labels_):
        # Indices of rectangles in the current cluster
        indices = np.where(clustering.labels_ == label)[0]
        
        # Coordinates for the merged rectangle
        x_min = min(rectangles[i][0] for i in indices)
        y_min = min(rectangles[i][1] for i in indices)
        x_max = max(rectangles[i][0] + rectangles[i][2] for i in indices)
        y_max = max(rectangles[i][1] + rectangles[i][3] for i in indices)
        
        # Append the merged rectangle to the list
        merged_rectangles.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
    return merged_rectangles

#========================================輔助函數========================================#

# 8. 計算輪廓周長 
def contour_perimeter(contour):
    perimeter = 0
    for i in range(len(contour)):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % len(contour)]
        perimeter += ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return perimeter

# 9. 計算邊界矩陣
def boundingRect(contour):
    
    x_coords = [point[1] for point in contour]
    y_coords = [point[0] for point in contour]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    return x_min, y_min, x_max - x_min, y_max - y_min

#========================================棄用函數========================================#

'''

# import cv2
# sys.path.append("C:/Users/a1234/Desktop/python/EAN13-barcode/2_locate_barcode/2_3_get_contour_angles")
# import get_contour_angles_2_3

# # 10. 繪製矩形
# def draw_rectangle(image, x, y, w, h, color, thickness):
#     # 上邊
#     image[y:y+thickness, x:x+w] = color
#     # 下邊
#     image[y+h-thickness:y+h, x:x+w] = color
#     # 左邊
#     image[y:y+h, x:x+thickness] = color
#     # 右邊
#     image[y:y+h, x+w-thickness:x+w] = color

#     return image

# # 11. 霍夫轉換
# def custom_hough_lines(image, rho_resolution=1, theta_resolution=np.pi/180, threshold=400):
#     """
#     Implement the Hough Transform to find lines in a binary image.

#     Parameters:
#     - image: 2D numpy array (binary image)
#     - rho_resolution: resolution of rho in pixels
#     - theta_resolution: resolution of theta in radians
#     - threshold: accumulator threshold parameter (minimum vote to consider a line)

#     Returns:
#     - lines: List of (rho, theta) tuples that represent detected lines
#     """
#     # Get image dimensions
#     height, width = image.shape
    
#     # Create the accumulator
#     max_rho = int(math.sqrt(height ** 2 + width ** 2))
#     rho_range = np.arange(-max_rho, max_rho, rho_resolution)
#     theta_range = np.arange(0, np.pi, theta_resolution)
#     accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=int)
    
#     # Coordinates of non-zero pixels (edge points)
#     y_idxs, x_idxs = np.nonzero(image)
    
#     # Perform the Hough Transform
#     for x, y in zip(x_idxs, y_idxs):
#         for theta_idx, theta in enumerate(theta_range):
#             rho = int(x * np.cos(theta) + y * np.sin(theta))
#             rho_idx = np.argmin(np.abs(rho_range - rho))
#             accumulator[rho_idx, theta_idx] += 1

#     # Find lines that pass the threshold
#     lines = []
#     rho_idxs, theta_idxs = np.where(accumulator >= threshold)
#     for rho_idx, theta_idx in zip(rho_idxs, theta_idxs):
#         rho = rho_range[rho_idx]
#         theta = theta_range[theta_idx]
#         lines.append((rho, theta))

#     return lines

# # 12. 統計出現最多次的角度
# def find_dominant_angle(lines):
#     """
#     Find the dominant angle from a list of lines in (rho, theta) format.
    
#     Parameters:
#     - lines: List of tuples, each containing (rho, theta)
    
#     Returns:
#     - dominant_angle: The angle (in degrees) that appears most frequently in the list
#     """
#     # Extract the theta values and convert them to degrees
#     theta_values = [math.degrees(theta) for rho, theta in lines]
    
#     # Count the occurrences of each unique angle
#     angle_count = Counter(theta_values)
    
#     # Find the most common angle
#     dominant_angle, _ = angle_count.most_common(1)[0]
    
#     return dominant_angle

# # 13. 取得輪廓角度
# def get_contour_details(filtered_contours):
#     angles = []

#     # Compute the minimum area rectangle for the contour
#     rects = get_contour_angles_2_3.custom_minAreaRect(filtered_contours)
    
#     # The angle of rotation of the rectangle
#     angles = [rect[2] for rect in rects]
    
#     angle_degrees = [angle * (180 / np.pi) for angle in angles]

#     print(angle_degrees)

#     mean_angle = sum(angles)/len(angles)

#     # Correct the angle values
#     if mean_angle < -45:
#         mean_angle = -(90 + mean_angle)
#     else:
#         mean_angle = -mean_angle
    
#     return mean_angle

# # 14. 繪製contours(檢查用)
# def draw_custom_contours(image, contours, color=(255, 255, 255), thickness=1):
#     """
#     Draw contours on the given image.

#     Parameters:
#         image (numpy.ndarray): The image on which to draw the contours.
#         contours (list): List of contours, where each contour is a list of points (x, y).
#         color (tuple): The color to use for the contours (default is white).
#         thickness (int): The thickness of the lines used to draw the contours (default is 2).
    
#     Returns:
#         numpy.ndarray: The image with the contours drawn.
#     """
#     for contour in contours:
#         for x, y in contour:
#             cv2.circle(image, (y, x), thickness, color, -1)  # 在每個輪廓點繪製一個圓點
#     return image

# # 15. 繪製霍夫直線(檢查用)
# def draw_hough_lines(image, lines, color=(255, 0, 0), thickness=2):
#     """
#     Draw lines on the image.

#     Parameters:
#     - image: 3D numpy array (the input image on which to draw)
#     - lines: List of (rho, theta) tuples that represent detected lines
#     - color: Tuple representing the color of the lines to be drawn (default is red)
#     - thickness: Integer representing the thickness of the lines (default is 2)

#     Returns:
#     - image_with_lines: 3D numpy array (the image with lines drawn)
#     """
#     image_with_lines = np.copy(image)
#     if len(image_with_lines.shape) == 2:  # If the image is grayscale, convert to color
#         image_with_lines = cv2.cvtColor(image_with_lines, cv2.COLOR_GRAY2BGR)
    
#     for rho, theta in lines:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(image_with_lines, (x1, y1), (x2, y2), color, thickness)
    
#     return image_with_lines

# # 16. dfs輪廓追蹤
# import trace_contours_2_2
# def trace_contours(edge_image):
#     print("dfs輪廓追蹤")
#     rows, cols = edge_image.shape
#     visited = np.zeros((rows, cols), dtype=np.int32)
#     all_contours = []

#     for i in range(rows):
#         for j in range(cols):
#             if visited[i, j] == 0 and edge_image[i, j] == 255:
#                 contour = trace_contours_2_2.dfs(i, j, visited, edge_image)
#                 print(contour)
#                 all_contours.append(contour)

#     wb = Workbook()
#     ws = wb.active
#     ws.append(['Contour_ID', 'X', 'Y'])

#     # 遍歷all_contours，並將數據添加到工作表
#     for i, contour in enumerate(all_contours):
#         for x, y in contour:
#             ws.append([i, x, y])

#     wb.save('contours.xlsx')

#     return all_contours



# # 17. 稀釋
# def manual_dilate(image, kernel):
#     # 獲取圖像和核的尺寸
#     image_h, image_w = image.shape
#     kernel_h, kernel_w = kernel.shape
    
#     # 計算填充尺寸
#     pad_h = kernel_h // 2
#     pad_w = kernel_w // 2
    
#     # 填充圖像
#     padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
#     # 初始化輸出圖像
#     output_image = np.zeros_like(image)
    
#     # 進行膨脹操作
#     for i in range(image_h):
#         for j in range(image_w):
#             # 取得和kernel對應的圖像區塊
#             sub_image = padded_image[i:i+kernel_h, j:j+kernel_w]
#             # 應用膨脹操作（取最大值）
#             output_image[i, j] = np.max(sub_image * kernel)
            
#     return output_image

# # 18. 腐蝕
# def manual_erode(image, kernel, iterations=1):
#     # 獲取圖像和核的尺寸
#     image_h, image_w = image.shape
#     kernel_h, kernel_w = kernel.shape
    
#     # 計算填充尺寸
#     pad_h = kernel_h // 2
#     pad_w = kernel_w // 2
    
#     # 初始化輸出圖像
#     output_image = np.copy(image)
    
#     # 進行腐蝕操作
#     for iter in range(iterations):
#         # 填充圖像
#         padded_image = np.pad(output_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
#         # 更新輸出圖像
#         output_image = np.zeros_like(image)
        
#         for i in range(image_h):
#             for j in range(image_w):
#                 # 取得和kernel對應的圖像區塊
#                 sub_image = padded_image[i:i+kernel_h, j:j+kernel_w]
                
#                 # 應用腐蝕操作（取最小值）
#                 if np.all(sub_image >= kernel):
#                     output_image[i, j] = np.min(sub_image)
                    
#     return output_image

'''