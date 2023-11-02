# 映射表
ODD_PARITY_MAP = {
    (3, 2, 1, 1): 0,  
    (2, 2, 2, 1): 1,  
    (2, 1, 2, 2): 2,  
    (1, 4, 1, 1): 3,  
    (1, 1, 3, 2): 4,  
    (1, 2, 3, 1): 5,  
    (1, 1, 1, 4): 6,  
    (1, 3, 1, 2): 7,  
    (1, 2, 1, 3): 8,  
    (3, 1, 1, 2): 9,  
}

EVEN_PARITY_MAP = {
    (1, 1, 2, 3): 0,
    (1, 2, 2, 2): 1,
    (2, 2, 1, 2): 2,  
    (1, 1, 4, 1): 3,  
    (2, 3, 1, 1): 4,  
    (1, 3, 2, 1): 5,  
    (4, 1, 1, 1): 6,  
    (2, 1, 3, 1): 7,  
    (3, 1, 2, 1): 8,  
    (2, 1, 1, 3): 9,  
}

# 1. 寛度轉數字
def width_to_number(stripe_widths):
    stripe_tuple = tuple(stripe_widths)
    
    # 原始查找
    odd_number = ODD_PARITY_MAP.get(stripe_tuple, None)
    if odd_number is not None:
        return odd_number, 'O'  # 'O' 表示 Odd parity
    
    even_number = EVEN_PARITY_MAP.get(stripe_tuple, None)
    if even_number is not None:
        return even_number, 'E'  # 'E' 表示 Even parity
    
    # 若找不到則開始“猜測”
    total = sum(stripe_tuple)
    possible_results = []

    if total == 6:
        # 對每個數字+1後重新尋找
        for i in range(4):
            adjusted_tuple = list(stripe_tuple)
            adjusted_tuple[i] += 1
            adjusted_tuple = tuple(adjusted_tuple)
            
            odd_number = ODD_PARITY_MAP.get(adjusted_tuple, None)
            if odd_number is not None:
                possible_results.append((odd_number, 'O'))
            
            even_number = EVEN_PARITY_MAP.get(adjusted_tuple, None)
            if even_number is not None:
                possible_results.append((even_number, 'E'))

    if total == 8:
        # 對每個數字-1後重新尋找
        for i in range(4):
            adjusted_tuple = list(stripe_tuple)
            adjusted_tuple[i] -= 1
            adjusted_tuple = tuple(adjusted_tuple)
            
            odd_number = ODD_PARITY_MAP.get(adjusted_tuple, None)
            if odd_number is not None:
                possible_results.append((odd_number, 'O'))
            
            even_number = EVEN_PARITY_MAP.get(adjusted_tuple, None)
            if even_number is not None:
                possible_results.append((even_number, 'E'))

    if possible_results:
        # 這裡返回所有可能的結果，你也可以選擇只返回第一個或某個特定的結果
        return possible_results

    return None, None  # 如果都找不到，也無法猜測
