#========================================棄用函數========================================#

'''
# # 1. 繪製掃描線

# def draw_line(img, start, end, color):
#     x1, y1 = start
#     x2, y2 = end
#     dx = x2 - x1
#     dy = y2 - y1

#     is_steep = abs(dy) > abs(dx)

#     if is_steep:
#         x1, y1 = y1, x1
#         x2, y2 = y2, x2

#     swapped = False
#     if x1 > x2:
#         x1, x2 = x2, x1
#         y1, y2 = y2, y1
#         swapped = True

#     dx = x2 - x1
#     dy = y2 - y1

#     error = int(dx / 2.0)
#     ystep = 1 if y1 < y2 else -1

#     y = y1
#     points = []
#     for x in range(x1, x2 + 1):
#         coord = (y, x) if is_steep else (x, y)
#         points.append(coord)
#         error -= abs(dy)
#         if error < 0:
#             y += ystep
#             error += dx

#     if swapped:
#         points.reverse()
    
#     for point in points:
#         if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0]:
#             img[point[1], point[0]] = color
#         else:
#             # Handle the case where the index is out of bounds
#             # For example, you could skip this iteration with 'continue'
#             continue

#     return img

'''