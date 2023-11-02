import sys
import cv2
import time
from itertools import product
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append("C:/Users/a1234/Desktop/python/EAN13-barcode/1_preprocess_image")
sys.path.append("C:/Users/a1234/Desktop/python/EAN13-barcode/2_locate_barcode")
sys.path.append("C:/Users/a1234/Desktop/python/EAN13-barcode/3_generate_scanlines")
sys.path.append("C:/Users/a1234/Desktop/python/EAN13-barcode/4_convert_to_numbers")
sys.path.append("C:/Users/a1234/Desktop/python/EAN13-barcode/5_decode_marks")
sys.path.append("C:/Users/a1234/Desktop/python/EAN13-barcode/9_process_image_at_angle")
import preprocess_image_1
import locate_barcode_2
import generate_scanlines_3
import convert_to_numbers_4
import decode_marks_5
import process_image_at_angle_9

#========================================檢查用代碼========================================#
# # 顯示影像
# cv2.imshow('image_name', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 顯示掃碼數據點
# print(f"寬度列表中有 {len(numbers)} 個數據點")
# print(numbers)

# starttime = time.time()
# endtime = time.time()
# print(f"圖像預處理耗費{endtime - starttime}秒")

#==========================================母函數==========================================#

# 1. 圖像預處理
def preprocess_image(image, block_size, c, n):
    
    # 圖像轉為灰階
    gray = preprocess_image_1.custom_cvtColor(image)
    
    # 獲取原始圖片的尺寸
    rows, cols = gray.shape

    # 預設寬度和高度為原始值
    new_width = cols
    new_height = rows

    # 檢查圖片寬度是否為 600
    if cols != 600:
        
        # 設置新寬度為 600
        new_width = 600
    
        # 計算新高度以保持等比例放大
        new_height = 800

    # 如果圖片過大、過小，調整大小 
    resized_gray = preprocess_image_1.bilinear_resize(gray, new_width, new_height)
    
    # 適應性二值化
    thresh = preprocess_image_1.adaptive_thresholding_center(resized_gray, block_size, c, n)

    return thresh

# 2. 條碼定位
def locate_barcode(thresh):

    # 邊緣檢測
    edge_image = locate_barcode_2.canny_edge_detection(thresh)
    
    # 獲取 edge_image 的尺寸
    h, w = edge_image.shape[:2]

    # 輪廓追蹤
    contours = locate_barcode_2.suzuki_contour_tracing(edge_image)

    # 過濾輪廓
    filtered_contours = locate_barcode_2.filter_contours(contours)

    # 儲存邊界矩形
    rectangles = []

    for contour in filtered_contours:
        x, y, w, h = locate_barcode_2.boundingRect(contour)
        rectangles.append((x, y, w, h))

    merged_rectangles = locate_barcode_2.merge_rectangles(rectangles)

    # 選擇面積最大的矩形
    large_rectangle = max(merged_rectangles, key=lambda r: r[2] * r[3])

    # x、y為矩形左下角的點
    x, y, w, h = large_rectangle

    return thresh, x, y, w, h

# 3. 掃描線生成函數
def generate_scanlines(image, x, y, w, h):

    # 初始化掃描線列表
    scanlines = []
    
    # 決定掃描線的數量
    num_scanlines = 40
    
    # 計算中心點的垂直位置
    center_y = y + h // 2
    
    # 設定一個垂直範圍，在這個範圍內生成掃描線
    vertical_range =  h // 2  # 範圍設為高度的1/2
    start_y = center_y - vertical_range // 2
    
    # 垂直間距
    spacing = vertical_range // (num_scanlines - 1)
    
    # 創建一個用於視覺化的影像副本
    vis_image = image.copy()
    
    for i in range(num_scanlines):
        current_y = start_y + i * spacing
        # 確保 current_y 在有效範圍內
        if current_y >= image.shape[0]:
            break
        
        height, width = image.shape[:2]

        start_point = (x - 10, current_y)
        end_point = (x + w + 10, current_y)
        
        # Modify the coordinates while keeping them within the bounds
        new_end_x = min(x + width, end_point[0])
        new_start_x = max(0, start_point[0])

        # Create new tuples
        start_point = (new_start_x, start_point[1])
        end_point = (new_end_x, end_point[1])

        # 初始化掃描線像素列表
        scanline_pixels = []
        
        for j in range(start_point[0], end_point[0]):
            if 0 <= current_y < image.shape[0] and 0 <= j < image.shape[1]:
                # 讀取像素值並添加到列表
                pixel_value = image[current_y, j]

                scanline_pixels.append(pixel_value)
            else:
                continue

        # 將這條掃描線的像素值列表添加到掃描線列表
        scanlines.append(scanline_pixels)

    return scanlines

# 4. 將條碼轉換為數字
def convert_to_numbers(scanlines):
    
    starttime = time.time()

    numbers = []  # 用來存儲轉換後的數字或其他形式的信息
    
    for scanline in scanlines:
        widths = []  # 用來存儲單個掃描線中的黑白條紋寬度
        count = 1  # 用來計數相同顏色的連續像素
        
        # 遍歷掃描線中的每一個像素
        for i in range(1, len(scanline)):
            if scanline[i] == scanline[i - 1]:
                # 如果與前一個像素的顏色相同，則計數加一
                count += 1
            else:
                # 如果顏色變了，則將計數添加到寬度列表中
                widths.append(count)
                count = 1  # 重置計數
                
        # 處理最後一個像素
        widths.append(count)
        
        # 新增的判斷式：檢查長度是否為48
        if len(widths) != 61:
            continue  # 跳過這個列表，繼續處理下一個掃描線

        # 找尋前、後、中央導航線
        center_idx = len(widths) // 2

        # 移除多餘寬度信息
        del widths[center_idx - 2 : center_idx + 3]  # 移除中央導航線
        del widths[:4] 
        del widths[-4:]

        # 將這一行的寬度信息添加到主列表中
        numbers.append(widths)
    
    # 將寛度轉換為比例關係、四捨五入
    avg_numbers = convert_to_numbers_4.average_and_process_numbers(numbers)
    
    endtime = time.time()

    print(f"將條碼轉換為數字耗費{endtime - starttime}秒")

    return avg_numbers

# 5. 標記解碼
def decode_marks(numbers):

    decoded_numbers = []  # 存儲解碼後數字
    parities = []  # 存儲奇偶性資訊

    i = 0
    index = 0  # 用於追踪當前位置
    while i < len(numbers):

        stripe_widths = numbers[i:i+4]
        result = decode_marks_5.width_to_number(stripe_widths)
        
        if result is not None:
            if isinstance(result, list):  # 檢查是否返回了多個可能選項
                if index >= len(numbers)//4 - 6:  # 檢查是否在後六個位置
                    result = [x for x in result if x[1] == 'O']  # 只留下'O'的結果

                decoded_numbers.append([item[0] for item in result])
                parities.append([item[1] for item in result])
            else:
                number, parity = result
                decoded_numbers.append(number)
                parities.append(parity)
        
        i += 4
        index += 1  # 更新當前位置

    return decoded_numbers, parities

# 6. 確認校驗碼
def find_first_digit(decoded_numbers, parities):

    # 第一碼的對應表
    FIRST_DIGIT_PARITY_MAP = {
        'OOOOOO': 0,
        'OOEOEE': 1,
        'OOEEOE': 2,
        'OOEEEO': 3,
        'OEOOEE': 4,
        'OEEOOE': 5,
        'OEEEOO': 6,
        'OEOEOE': 7,
        'OEOEEO': 8,
        'OEEOEO': 9,
    }

    # 反轉奇偶性
    def flip_parity(p):
        return 'O' if p == 'E' else 'E'

    # 尋找第一碼
    def try_find_first_digit(decoded_numbers, parities):
        guess_indices = [i for i, p in enumerate(parities[:6]) if isinstance(p, list)]
        all_possible_combinations = product(*[parities[i] for i in guess_indices])
        
        for combination in all_possible_combinations:
            temp_parities = parities[:6].copy()
            for i, value in zip(guess_indices, combination):
                temp_parities[i] = value
            
            first_six_parities = ''.join(temp_parities)
            first_digit = FIRST_DIGIT_PARITY_MAP.get(first_six_parities, None)
            
            if first_digit is not None:
                decoded_numbers.insert(0, first_digit)
                return decoded_numbers, True
        return None, False

    # 先嘗試照原排序找出第一碼
    result, success = try_find_first_digit(decoded_numbers, parities)
    if success:
        return result, True

    # 如果找不到，反轉排序後再嘗試一次(圖片旋轉後可能上下顛倒)
    reversed_decoded_numbers = decoded_numbers[::-1]
    flipped_parities = [flip_parity(p) if p in ['O', 'E'] else p for p in parities[::-1]]
    result, success = try_find_first_digit(reversed_decoded_numbers, flipped_parities)
    if success:
        return result, True

    return None, False

# 7. 檢驗解碼結果
def validate_checksum_ean13(decoded_numbers):
    
    # 找出包含猜測答案（列表）的位置
    guess_indices = [i for i, num in enumerate(decoded_numbers) if isinstance(num, list)]
    
    # 生成所有可能的數字組合
    all_possible_combinations = list(product(*[decoded_numbers[i] for i in guess_indices]))
    
    # 創建一個包含所有未試過的數字（0-9）的列表
    untried_numbers = {i: set(range(10)) for i in guess_indices}
    
    for i in guess_indices:
        untried_numbers[i] -= set(decoded_numbers[i])
    
    # 驗證猜測的答案是否正確
    for combination in all_possible_combinations:
        temp_decoded_numbers = decoded_numbers.copy()
        for i, value in zip(guess_indices, combination):
            temp_decoded_numbers[i] = value
        
        checksum = 0
        for i, num in enumerate(reversed(temp_decoded_numbers[:-1])):
            if i % 2 == 0:
                checksum += num * 3
            else:
                checksum += num
        
        # 找到最接近的 10 的倍數
        nearest_ten = 10 * ((checksum + 9) // 10)
        
        # 計算校驗碼
        calculated_checksum = nearest_ten - checksum
        
        if calculated_checksum == temp_decoded_numbers[-1]:
            return True, temp_decoded_numbers
    
    # 使用其它未試過的數字尋找校驗碼
    for i in guess_indices:
        all_possible_combinations.extend(product(*[list(untried_numbers[i])]))
    
    for combination in all_possible_combinations:
        temp_decoded_numbers = decoded_numbers.copy()
        for i, value in zip(guess_indices, combination):
            temp_decoded_numbers[i] = value

        checksum = 0
        for i, num in enumerate(reversed(temp_decoded_numbers[:-1])):
            if i % 2 == 0:
                checksum += num * 3
            else:
                checksum += num

        nearest_ten = 10 * ((checksum + 9) // 10)
        calculated_checksum = nearest_ten - checksum

        if calculated_checksum == temp_decoded_numbers[-1]:

            return True, temp_decoded_numbers

    return False, None

# 8. 輸出結果函數(已完成)
def output_result(decoded_numbers):
    print(decoded_numbers)

# 9. 單次解碼
def process_image_at_angle(angle, image_path):
    
    print(f"Trying angle {angle}...")
    
    image = cv2.imread(image_path)

    if image is None:
        print(f"無法從 {image_path}載入圖片")
        return

    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = process_image_at_angle_9.getRotationMatrix2D(center, angle, 1)
    rotated_image = process_image_at_angle_9.warpAffine(image, M, (image.shape[0], image.shape[1]))

    image_find_barcode = preprocess_image(rotated_image,  block_size = 200, c = 18, n = 10)
    located_image, x, y, w, h = locate_barcode(image_find_barcode)
    scanlines = generate_scanlines(located_image, x, y, w, h)
    numbers = convert_to_numbers(scanlines)
    
    if not numbers:
        return None

    decoded_numbers, parities = decode_marks(numbers)
    decoded_numbers13, success = find_first_digit(decoded_numbers, parities)
    
    if decoded_numbers13 is None:
        return None

    is_valid, true_numbers = validate_checksum_ean13(decoded_numbers13)
    
    if is_valid:
        return true_numbers
    else:
        return None

#=========================================主程式=========================================#

def main():
    
    image_path = "C:/Users/a1234/Desktop/python/EAN13-barcode/test14.jpg"

    # 創建一個字典來保存每個 result 和其出現次數
    results_counter = Counter()
    
    # 創建一個 ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        
        # 為每個角度啟動一個進程
        angles = range(0, 360, 15)
        futures = {executor.submit(process_image_at_angle, angle, image_path): angle for angle in angles}

        # 收集結果（這部分會等待所有進程完成）
        for future in as_completed(futures):
            angle = futures[future]
            try:
                result = future.result()
                if result is not None:  # 檢查 result 是否為 None
                    results_counter[tuple(result)] += 1  # 將列表轉換為元組，然後累加結果次數
                
            except Exception as e:
                print(f"Error processing angle {angle}: {e}")

    # 找出出現次數最多的 result
    if results_counter:
        most_common_result, _ = results_counter.most_common(1)[0]
        output_result(list(most_common_result))  # 將元組轉回列表
    else:
        print("沒有有效的結果")

if __name__ == '__main__':
    main()
