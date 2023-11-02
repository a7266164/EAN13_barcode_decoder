import numpy as np

# 1. 尋找旋轉矩陣
def getRotationMatrix2D(center, angle, scale):
    angle = np.deg2rad(angle)  # Convert angle to radians
    alpha = scale * np.cos(angle)
    beta = scale * np.sin(angle)
    
    # Create the rotation matrix
    M = np.array([
        [alpha, -beta, (1 - alpha) * center[0] + beta * center[1]],
        [beta, alpha, -beta * center[0] + (1 - alpha) * center[1]]
    ])
    
    return M

# 2. 旋轉圖片
def warpAffine(image, M, dsize):
    h, w = dsize
    output = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
    
    for y in range(h):
        for x in range(w):
            # Apply the inverse transformation for each point in the output image
            src_point = np.array([x, y, 1])
            dst_point = M.dot(src_point)
            
            x_dst, y_dst = int(dst_point[0]), int(dst_point[1])
            
            # Check if the point falls within the boundaries of the source image
            if 0 <= x_dst < image.shape[1] and 0 <= y_dst < image.shape[0]:
                output[y, x] = image[y_dst, x_dst]
                
    return output