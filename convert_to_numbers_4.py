# 優化完畢

# 1. 將寛度轉換為比例關係、四捨五入
def average_and_process_numbers(numbers_list):
    if not numbers_list:
        return []
    
    avg_list = []
    for i in range(len(numbers_list[0])):  # Assume all sublists are of the same length
        avg = sum(numbers[i] for numbers in numbers_list) / len(numbers_list)
        avg_rounded = round(avg, 3)  # Round to 3 decimal places
        avg_list.append(avg_rounded)
    
    new_list = []
    for i in range(0, len(avg_list), 4):
        unit = avg_list[i:i+4]
        unit_sum = sum(unit)
        divisor = unit_sum / 7.0  # Add the four numbers and divide by 7
        
        normalized_unit = [round(x / divisor) for x in unit]  # Divide each number by the result and round
        new_list.extend(normalized_unit)
        
    return new_list

#========================================棄用函數========================================#

'''

# # 2. 寛度正規化
# def normalize_with_navigation_lines(widths):
#     # 前導航線、後導航線和中央導航線的寬度
#     front_nav = widths[:3]
#     back_nav = widths[-3:]
#     center_idx = len(widths) // 2
#     center_nav = widths[center_idx - 1 : center_idx + 2]
    
#     print(front_nav)
#     print(back_nav)
#     print(center_nav)
#     # 計算導航線的平均寬度
#     avg_nav_width = sum(front_nav + back_nav + center_nav) / 9.0
    
#     # 使用導航線的平均寬度作為基準來計算其他 bar 的相對寬度
#     normalized_widths = [w / avg_nav_width for w in widths]
    
#     return normalized_widths

# # 3. 將寛度轉換為比例關係、四捨五入(全部平均版)

# def process_avg_numbers_v2(avg_numbers):
#     new_list = []
#     total_sum = sum(avg_numbers)  # 計算整個列表的總和
#     divisor = total_sum / 84.0  # 將總和除以84
    
#     normalized_unit = [round(x / divisor) for x in avg_numbers]  # 將每個數字分別除以該結果後，四捨五入
#     new_list.extend(normalized_unit)

#     return new_list

'''