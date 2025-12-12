import cv2
import numpy as np
import os
from datetime import datetime
from scipy.spatial import cKDTree  # 用于加速最近邻搜索
# =============================================================================
# 通用辅助函数
# =============================================================================

def _save_step_image(img, folder, filename):
    """(内部辅助) 保存过程图片，自动处理浮点/uint8"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    if img.dtype != np.uint8:
        # 视觉化: 如果有负数(拉普拉斯层)，偏移显示
        if img.min() < 0:
            visual = np.clip(img * 2 + 0.5, 0, 1) 
            save_img = (visual * 255).astype(np.uint8)
        else:
            save_img = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        save_img = img
    cv2.imwrite(path, save_img)

def _apply_freq_filter_channel(img_channel, mask):
    """(内部辅助) 单通道频域滤波核心逻辑"""
    f = np.fft.fft2(img_channel.astype(np.float32))
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.clip(np.abs(img_back), 0, 255).astype(np.uint8)

# =============================================================================
# 基础调整算法
# =============================================================================

def to_gray(image, **kwargs):
    if len(image.shape) == 3:
        res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR) 
    return image

def gamma_correction(image, gamma=1.0, **kwargs):
    gamma = float(gamma)
    invGamma = 1.0 / (gamma + 1e-6)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def color_saturation_boost(image, scale=1.0, **kwargs):
    scale = float(scale)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# =============================================================================
# 增强与锐化
# =============================================================================

def clahe_enhance(image, clip_limit=2.0, **kwargs):
    clip_limit = float(clip_limit)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def unsharp_mask(image, sigma=2.0, amount=1.5, **kwargs):
    sigma = float(sigma)
    amount = float(amount)
    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, gaussian, -amount, 0)

def morph_contrast_enhance(image, kernel_size=15, **kwargs):
    k_size = int(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    tophat = cv2.morphologyEx(l, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(l, cv2.MORPH_BLACKHAT, kernel)
    temp = cv2.add(l, tophat)
    l_enhanced = cv2.subtract(temp, blackhat)
    lab_merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

# =============================================================================
# 降噪算法
# =============================================================================

def bilateral_filter_denoise(image, d=9, sigma=75, **kwargs):
    return cv2.bilateralFilter(image, int(d), float(sigma), float(sigma))

def nlm_denoise_colored(image, h=10, h_color=10, templateWindowSize=7, searchWindowSize=21, **kwargs):
    """
    NLM 强力降噪 (开放所有参数)
    """
    return cv2.fastNlMeansDenoisingColored(image, None, float(h), float(h_color), int(templateWindowSize), int(searchWindowSize))

def chroma_denoise(image, kernel_size=21, **kwargs):
    k = int(kernel_size) | 1 # 确保奇数
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.medianBlur(a, k)
    b = cv2.medianBlur(b, k)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# =============================================================================
# 去雾
# =============================================================================

def _get_dark_channel(img, size):
    b, g, r = cv2.split(img)
    min_img = cv2.min(cv2.min(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_img, kernel)

def _get_atmospheric_light(img, dark_channel):
    h, w = img.shape[:2]
    img_size = h * w
    num_pixels = int(max(img_size * 0.001, 1))
    dark_vec = dark_channel.reshape(img_size)
    img_vec = img.reshape(img_size, 3)
    indices = dark_vec.argsort()[::-1][:num_pixels]
    atms_sum = np.zeros(3)
    for ind in indices: atms_sum += img_vec[ind]
    return atms_sum / num_pixels

def _guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    return mean_a * I + mean_b

def dehaze_dcp(image, omega=0.95, **kwargs):
    omega = float(omega)
    I = image.astype('float64') / 255.0
    dark = _get_dark_channel(I, 15)
    A = _get_atmospheric_light(I, dark)
    norm_I = I / A
    t = 1 - omega * _get_dark_channel(norm_I, 15)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float64') / 255.0
    t_refined = _guided_filter(gray, t, 40, 1e-3)
    J = np.empty_like(I)
    for i in range(3):
        J[:, :, i] = (I[:, :, i] - A[i]) / np.maximum(t_refined, 0.1) + A[i]
    return np.clip(J * 255, 0, 255).astype('uint8')
# -------------------------------------------------------
# 新增：空间自适应去雾 (中心/边缘独立控制)
# -------------------------------------------------------
def _create_spatial_omega_map(shape, w_center, w_edge, radius):
    """(内部辅助) 创建高斯分布的权重图"""
    rows, cols = shape[:2]
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    
    # 归一化距离 (以短边为基准)
    min_dim = min(rows, cols)
    dist_sq = ((x - ccol)**2 + (y - crow)**2) / ((min_dim/2)**2)
    
    # 高斯衰减
    sigma_sq = max(radius, 0.1) ** 2
    mask = np.exp(-dist_sq / (2 * sigma_sq))
    
    # 混合
    omega_map = w_edge + (w_center - w_edge) * mask
    return np.clip(omega_map, 0.01, 0.99)

def dehaze_dcp_spatial(image, omega_center=0.95, omega_edge=0.6, radius=0.6, **kwargs):
    w_cen = float(omega_center)
    w_edge = float(omega_edge)
    rad = float(radius)
    
    I = image.astype('float64') / 255.0
    dark = _get_dark_channel(I, 15)
    A = _get_atmospheric_light(I, dark)
    
    norm_I = I / A
    
    # 生成空间变化的 Omega
    omega_map = _create_spatial_omega_map(image.shape, w_cen, w_edge, rad)
    
    # 计算透射率 t (使用矩阵乘法)
    raw_t = 1 - omega_map * _get_dark_channel(norm_I, 15)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float64') / 255.0
    t_refined = _guided_filter(gray, raw_t, 40, 1e-3)
    
    J = np.empty_like(I)
    for i in range(3):
        J[:, :, i] = (I[:, :, i] - A[i]) / np.maximum(t_refined, 0.1) + A[i]
    
    return np.clip(J * 255, 0, 255).astype('uint8')

# =============================================================================
# 高级：拉普拉斯金字塔融合 (参数完全开放)
# =============================================================================

def _process_ch_laplacian_advanced(channel, ch_name, params, output_dir):
    """
    params 结构:
    - base_h, base_temp, base_search (底图NLM)
    - detail_d, detail_sigma_c, detail_sigma_s (细节双边)
    - w_l0, w_l1 (融合权重)
    """
    
    # 1. 底图去噪 (NLM)
    ch_uint8 = np.clip(channel * 255, 0, 255).astype(np.uint8)
    
    base_h = params.get('base_h', 5.0)
    base_temp = int(params.get('base_temp', 7))
    base_search = int(params.get('base_search', 21))
    
    base_uint8 = cv2.fastNlMeansDenoising(ch_uint8, None, h=base_h, 
                                          templateWindowSize=base_temp, 
                                          searchWindowSize=base_search)
    base_float = base_uint8.astype(np.float32) / 255.0
    
    if output_dir: _save_step_image(base_float, output_dir, f"{ch_name}_01_Base.jpg")

    # 2. 金字塔分解
    G = channel.copy()
    G_down0 = cv2.pyrDown(G)
    L0 = cv2.subtract(G, cv2.pyrUp(G_down0, dstsize=(G.shape[1], G.shape[0])))
    
    G_down1 = cv2.pyrDown(G_down0)
    L1 = cv2.subtract(G_down0, cv2.pyrUp(G_down1, dstsize=(G_down0.shape[1], G_down0.shape[0])))
    
    if output_dir:
        _save_step_image(L0, output_dir, f"{ch_name}_02_L0_Raw.jpg")

    # 3. 细节层去噪 (双边滤波)
    # L0 和 L1 都包含负数，双边滤波通常针对0-255或0-1的正数
    # 策略：移位 -> 滤波 -> 移回
    
    d = int(params.get('detail_d', 0))
    sc = float(params.get('detail_sigma_c', 0))
    ss = float(params.get('detail_sigma_s', 0))
    
    def denoise_layer(layer_float):
        # 如果参数为0，跳过去噪
        if d <= 0 and sc <= 0: return layer_float
        
        # 偏移到 0-1 范围 (假设原范围约 -0.5 ~ 0.5)
        temp = np.clip(layer_float + 0.5, 0, 1).astype(np.float32)
        filtered = cv2.bilateralFilter(temp, d, sc, ss)
        return filtered - 0.5

    L0_proc = denoise_layer(L0)
    L1_proc = denoise_layer(L1) # 通常也对L1做同样处理，或不做处理
    
    if output_dir and (d > 0 or sc > 0):
        _save_step_image(L0_proc, output_dir, f"{ch_name}_03_L0_Denoised.jpg")

    # 4. 融合
    w0 = params.get('w_l0', 1.0)
    w1 = params.get('w_l1', 1.0)
    
    L1_up = cv2.resize(L1_proc, (channel.shape[1], channel.shape[0]))
    fused = base_float + (w0 * L0_proc) + (w1 * L1_up)
    fused = np.clip(fused, 0, 1)
    
    if output_dir: _save_step_image(fused, output_dir, f"{ch_name}_04_Fused.jpg")
    return fused

def laplacian_pyramid_fusion(image, **kwargs):
    # 1. 准备输出目录 (调试用)
    root_dir = "process_out"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    date_str = datetime.now().strftime("%Y%m%d")
    index = 1
    # 循环查找当前日期下可用的序号，确保不覆盖
    while True:
        sub_dir_name = f"{date_str}_{index}_lap_step_out"
        output_dir = os.path.join(root_dir, sub_dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            break
        index += 1
        
    # 2. 转为浮点数进行计算 (0.0 - 1.0)
    img_float = image.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)
    
    # 提取参数辅助函数
    def get_params(prefix):
        return {
            'base_h': kwargs.get(f'{prefix}_base_h', 5),
            'base_temp': kwargs.get(f'{prefix}_base_temp', 7),
            'base_search': kwargs.get(f'{prefix}_base_search', 21),
            'detail_d': kwargs.get(f'{prefix}_det_d', 0),
            'detail_sigma_c': kwargs.get(f'{prefix}_det_sc', 0),
            'detail_sigma_s': kwargs.get(f'{prefix}_det_ss', 0),
            'w_l0': kwargs.get(f'{prefix}_w_l0', 1.0),
            'w_l1': kwargs.get(f'{prefix}_w_l1', 1.0),
        }

    # 3. 分通道处理 (这里调用了你上面定义的 _process_ch_laplacian_advanced)
    # 注意：_process_ch_laplacian_advanced 返回的是 float32 (0.0-1.0)
    b_fused = _process_ch_laplacian_advanced(b, 'Blue', get_params('b'), output_dir)
    g_fused = _process_ch_laplacian_advanced(g, 'Green', get_params('g'), output_dir)
    r_fused = _process_ch_laplacian_advanced(r, 'Red', get_params('r'), output_dir)
    
    # 4. 合并通道
    merged = cv2.merge([b_fused, g_fused, r_fused])

    # 将 Float(0-1) 转换为 Uint8(0-255)
    # 强制转换为无符号8位整数 (astype uint8)
    result = np.clip(merged * 255, 0, 255).astype(np.uint8)
    
    return result


def grabcut_interactive(image, rect, iter_count=5, **kwargs):
    """
    交互式 GrabCut
    rect: (x, y, w, h) 用户框选的矩形坐标
    """
    if rect is None or rect[2] == 0 or rect[3] == 0:
        return image # 如果框无效，返回原图

    iter_count = int(iter_count)
    
    # 1. 创建掩码和模型
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 2. 执行 GrabCut (使用用户传入的 rect)
    # 增加 try-catch 防止 rect 越界导致崩溃
    try:
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print(f"GrabCut Error: {e}")
        return image

    # 3. 提取结果
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    
    return result

def morph_edge_detection(image, kernel_size=3, mode=0, **kwargs):
    """
    形态学边缘检测
    :param kernel_size: 结构元素大小 (必须奇数)，决定边缘粗细
    :param mode: 0=标准梯度(D-E), 1=外部边缘(D-I), 2=内部边缘(I-E)
    """
    k = int(kernel_size)
    if k < 1: k = 1
    if k % 2 == 0: k += 1 # 确保奇数

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    # 根据模式计算
    if mode == 0:
        # 标准形态学梯度 = 膨胀 - 腐蚀
        # 这是最常用的方法，边缘位于轮廓中心
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    
    elif mode == 1:
        # 外部边缘 = 膨胀 - 原图
        # 边缘会比物体本身稍微大一圈
        dilation = cv2.dilate(image, kernel)
        return cv2.subtract(dilation, image)
    
    elif mode == 2:
        # 内部边缘 = 原图 - 腐蚀
        # 边缘会包含在物体内部
        erosion = cv2.erode(image, kernel)
        return cv2.subtract(image, erosion)
        
    return image

def binary_threshold(image, thresh_val=127, method=0, **kwargs):
    """
    图像二值化处理
    :param thresh_val: 手动阈值 (0-255)
    :param method: 0=手动全局, 1=Otsu自动, 2=自适应(高斯)
    """
    # 1. 预处理：二值化必须在灰度图上进行
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    thresh_val = int(thresh_val)
    method = int(method)

    if method == 0:
        # --- 模式0: 手动全局阈值 ---
        # 超过 thresh_val 的像素设为 255 (白)，其余为 0 (黑)
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    
    elif method == 1:
        # --- 模式1: Otsu 大津法自动阈值 ---
        # 自动寻找双峰直方图的最佳分割点，忽略传入的 thresh_val
        # 返回的 _ 是计算出的最佳阈值
        best_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Otsu 自动计算的阈值为: {best_val}")
        
    elif method == 2:
        # --- 模式2: 自适应阈值 (Adaptive Gaussian) ---
        # 适用于光照不均。它计算每个像素邻域的加权平均值作为该像素的阈值。
        # blockSize=11 (邻域大小), C=2 (从计算均值中减去的常数)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    else:
        return image

    # 为了保持平台显示一致性，将单通道二值图转回 3通道 BGR
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

from scipy.spatial import cKDTree

def restoration_idw_external_mask(image, rect, mask_path, k_neighbors=5, **kwargs):
    """
    基于外部二值化掩码的 IDW 修复
    :param rect: (x, y, w, h) 原图上的选区
    :param mask_path: 外部二值化图片的路径 (字符串)
    :param k_neighbors: 最近邻数量
    """
    if rect is None or not mask_path:
        print("错误：选区无效或未选择掩码路径")
        return image
    
    # 路径清理（去除可能的引号等）
    mask_path = str(mask_path).strip('"').strip("'")
    
    if not os.path.exists(mask_path):
        print(f"错误：找不到文件 {mask_path}")
        return image
        
    x, y, w, h = rect
    
    # 1. 读取外部掩码 (强制转为灰度)
    # 使用 cv2.IMREAD_GRAYSCALE 确保读入单通道
    # 使用 imdecode 支持中文路径
    mask_full = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    
    if mask_full is None:
        print("错误：无法读取掩码图片")
        return image

    # 2. 安全检查：确保掩码和原图尺寸一致
    # 如果尺寸不一致，强制缩放掩码以匹配原图 (鲁棒性处理)
    if mask_full.shape[:2] != image.shape[:2]:
        print(f"警告：掩码尺寸 {mask_full.shape} 与原图 {image.shape} 不一致，已自动调整。")
        mask_full = cv2.resize(mask_full, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 3. 提取对应的 ROI
    img_roi = image[y:y+h, x:x+w]
    mask_roi = mask_full[y:y+h, x:x+w]
    
    if img_roi.size == 0: return image

    # 4. 识别目标点与参考点
    # 假设：掩码中 > 127 (白) 为需要修复的点，<= 127 (黑) 为参考点
    target_coords = np.argwhere(mask_roi > 127) # 白色：瑕疵
    valid_coords = np.argwhere(mask_roi <= 127) # 黑色：好点

    if len(valid_coords) == 0 or len(target_coords) == 0:
        print("区域内无需修复或无参考点")
        return image 

    # 5. KD-Tree 加速查找
    tree = cKDTree(valid_coords)
    k = int(k_neighbors)
    # 查询 k 个最近邻
    dists, indices = tree.query(target_coords, k=k, workers=-1)

    # 6. IDW (反距离加权) 计算
    epsilon = 1e-6
    weights = 1.0 / (dists + epsilon)
    
    if k == 1:
        weights = weights[:, np.newaxis]
        indices = indices[:, np.newaxis]

    weight_sum = np.sum(weights, axis=1, keepdims=True)
    norm_weights = weights / weight_sum

    # 获取颜色并加权
    # 注意：valid_coords 是 (y, x)
    valid_colors = img_roi[valid_coords[:, 0], valid_coords[:, 1]]
    neighbor_colors = valid_colors[indices] # (N_target, k, 3)
    
    interpolated = np.sum(norm_weights[:, :, np.newaxis] * neighbor_colors, axis=1)

    # 7. 填回 ROI
    img_roi[target_coords[:, 0], target_coords[:, 1]] = interpolated.astype(np.uint8)
    
    result = image.copy()
    result[y:y+h, x:x+w] = img_roi
    
    return result

def perspective_correction(image, points, target_width=0, target_height=0, **kwargs):
    """
    透视变换校正
    :param points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 四个点坐标
    :param target_width: 输出宽度 (0则自动计算)
    :param target_height: 输出高度 (0则自动计算)
    """
    if points is None or len(points) != 4:
        print("错误：必须选择 4 个点")
        return image

    pts1 = np.float32(points)
    
    # 如果用户没有指定宽高，根据选框的最大边长自动计算
    if target_width == 0 or target_height == 0:
        # 获取四个点
        (tl, tr, br, bl) = pts1
        
        # 计算宽度 (取上下边长的最大值)
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # 计算高度 (取左右边长的最大值)
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if target_width == 0: target_width = maxWidth
        if target_height == 0: target_height = maxHeight

    # 定义目标图的四个角 (左上, 右上, 右下, 左下)
    pts2 = np.float32([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ])

    # 生成变换矩阵并执行
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (int(target_width), int(target_height)))
    
    return dst

# =============================================================================
# 局部掩码生成 (魔棒)
# =============================================================================
def generate_local_mask(image, rect, points_relative, tolerance=20, **kwargs):
    """
    局部掩码生成 (支持多点):
    对每个种子点计算掩码范围，最后取并集。
    points_relative: 包含多个 (x, y) 元组的列表
    """
    if rect is None or not points_relative:
        return image

    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    
    # 初始化一个全黑的 ROI 掩码
    accumulated_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
    tol = float(tolerance)

    # 【核心修改】遍历所有选中的点
    # 兼容性处理：如果只传了一个点(非列表)，转为列表
    if not isinstance(points_relative, list):
        points_relative = [points_relative]

    for (px, py) in points_relative:
        # 安全检查
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
            
        seed_color = roi[py, px]
        
        # 计算该点的上下限
        lower_bound = np.clip(seed_color - tol, 0, 255).astype(np.uint8)
        upper_bound = np.clip(seed_color + tol, 0, 255).astype(np.uint8)
        
        # 生成该点的掩码
        mask_i = cv2.inRange(roi, lower_bound, upper_bound)
        
        # 【关键】使用 bitwise_or 将新掩码合并到总掩码中 (取并集)
        accumulated_mask_roi = cv2.bitwise_or(accumulated_mask_roi, mask_i)
    
    # 放入全黑背景
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y:y+h, x:x+w] = accumulated_mask_roi
    
    return cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)

# =============================================================================
# 拉普拉斯金字塔：拆解功能 (提取保存 & 细节回注)
# =============================================================================

def laplacian_extract_save(image, levels=3, **kwargs):
    """
    提取拉普拉斯金字塔各层并保存为文件
    :param levels: 金字塔层数
    """
    # 1. 准备输出目录
    root_dir = "process"
    if not os.path.exists(root_dir): os.makedirs(root_dir)

    date_str = datetime.now().strftime("%Y%m%d")
    index = 1
    while True:
        # 文件夹命名：Extract_Laplacian
        sub_dir_name = f"{date_str}_{index}_Extract_Laplacian"
        output_dir = os.path.join(root_dir, sub_dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            break
        index += 1

    print(f"正在提取特征，保存至: {output_dir}")
    
    # 2. 转换数据类型
    current_img = image.astype(np.float32)
    
    for i in range(int(levels)):
        # 下采样
        down = cv2.pyrDown(current_img)
        # 上采样恢复尺寸
        up = cv2.pyrUp(down, dstsize=(current_img.shape[1], current_img.shape[0]))
        
        # 计算拉普拉斯层 (细节 = 原图 - 模糊图)
        laplacian = current_img - up
        
        # 3. 保存细节层
        # 关键技巧：拉普拉斯层包含负数。
        # 为了保存为可见且可逆的图片，我们使用“线性光”逻辑：将 0 映射为 128 (中性灰)
        # 保存公式: uint8 = float + 128
        save_img = np.clip(laplacian + 128.0, 0, 255).astype(np.uint8)
        
        filename = f"Level_{i}_HighFreq.png" # 使用 png 无损格式以保证精度
        cv2.imwrite(os.path.join(output_dir, filename), save_img)
        
        # 准备下一层
        current_img = down
        
    # 保存最后的残差层 (Base)
    base_img = np.clip(current_img, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"Level_{int(levels)}_Base.png"), base_img)
    
    print(f"提取完成，共 {int(levels)} 层细节 + 1 层底图")
    # 返回原图，不改变当前显示
    return image


def laplacian_inject_layer(image, layer_path, strength=1.0, **kwargs):
    """
    注入拉普拉斯特征层 (细节恢复)
    :param layer_path: 之前保存的 Level_x_HighFreq.png 文件路径
    :param strength: 注入强度 (支持负数，负数等于磨皮/模糊)
    """
    if not layer_path: return image
    layer_path = str(layer_path).strip('"').strip("'")
    if not os.path.exists(layer_path):
        print(f"错误：找不到文件 {layer_path}")
        return image
        
    # 读取细节层
    detail_img = cv2.imdecode(np.fromfile(layer_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if detail_img is None: return image
    
    # 尺寸检查与调整
    h, w = image.shape[:2]
    if detail_img.shape[:2] != (h, w):
        print(f"提示：特征层尺寸 {detail_img.shape[:2]} 与原图 {image.shape[:2]} 不一致，已自动缩放")
        detail_img = cv2.resize(detail_img, (w, h), interpolation=cv2.INTER_CUBIC)

    # 1. 还原细节数值
    # 载入 (0-255)，还原为 (-128 到 +127)
    # detail_float = detail_uint8 - 128.0
    detail_float = detail_img.astype(np.float32) - 128.0
    
    # 2. 叠加
    img_float = image.astype(np.float32)
    strength = float(strength)
    
    # Result = Base + Strength * Detail
    result = img_float + (detail_float * strength)
    
    # 3. 限制范围并返回
    return np.clip(result, 0, 255).astype(np.uint8)

def laplacian_inject_multilevel(image, anchor_path, weights_str="1.0, 0.5, 0.2", **kwargs):
    """
    多层拉普拉斯特征融合
    :param anchor_path: 特征文件夹下的任意一个文件 (用于定位目录)
    :param weights_str: 权重字符串，逗号分隔 (例如 "1.0, 0.5, 0") 分别对应 Level_0, Level_1...
    """
    if not anchor_path: return image
    anchor_path = str(anchor_path).strip('"').strip("'")
    if not os.path.exists(anchor_path): return image
    
    # 1. 获取目录和权重列表
    search_dir = os.path.dirname(anchor_path)
    try:
        # 将 "1.0, 0.5" 解析为 [1.0, 0.5]
        weights = [float(w.strip()) for w in weights_str.split(',')]
    except:
        print("错误：权重格式不正确，请使用英文逗号分隔的数字")
        return image

    print(f"开始多层融合，目标目录: {search_dir}")
    print(f"各层权重: {weights}")

    img_float = image.astype(np.float32)
    h, w = image.shape[:2]

    # 2. 遍历权重，寻找对应的层文件
    # weights[0] -> 寻找 Level_0_xxx.png
    # weights[1] -> 寻找 Level_1_xxx.png
    for level_idx, strength in enumerate(weights):
        if strength == 0: continue # 权重为0则跳过

        # 搜索匹配的文件名 (不区分 HighFreq 或 LoG，只要开头匹配 Level_X_)
        target_prefix = f"Level_{level_idx}_"
        found_file = None
        
        for fname in os.listdir(search_dir):
            if fname.startswith(target_prefix) and (fname.endswith(".png") or fname.endswith(".jpg")):
                # 排除 Base 层，我们通常只注入细节层，除非你想做图像重构
                if "Base" in fname: continue 
                found_file = os.path.join(search_dir, fname)
                break
        
        if not found_file:
            print(f"警告: 未找到 Level {level_idx} 的特征文件，跳过。")
            continue
            
        # 3. 读取并注入
        print(f"-> 正在注入: {os.path.basename(found_file)} (强度 {strength})")
        layer_img = cv2.imdecode(np.fromfile(found_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # 自动缩放尺寸以适应当前图片 (关键步骤：金字塔层通常比原图小)
        if layer_img.shape[:2] != (h, w):
            layer_img = cv2.resize(layer_img, (w, h), interpolation=cv2.INTER_CUBIC)
            
        # 还原数值 (-128 ~ 127) 并加权叠加
        layer_float = layer_img.astype(np.float32) - 128.0
        
        # 核心融合公式: Result += Layer * Weight
        img_float += layer_float * strength

    # 4. 输出
    return np.clip(img_float, 0, 255).astype(np.uint8)

# =============================================================================
# SWD 风格迁移 (新增)
# =============================================================================
def color_transfer_swd(image, ref_path, iter_count=30, proj_count=64, **kwargs):
    """
    Sliced Wasserstein Distance 色彩迁移
    :param ref_path: 参考风格图片的路径
    :param iter_count: 迭代次数 (建议 20-100)
    :param proj_count: 投影数量 (建议 50-200)
    """
    # 1. 检查并读取参考图片
    if not ref_path:
        print("错误：未提供参考图片路径")
        return image
    
    # 清理路径字符串
    ref_path = str(ref_path).strip('"').strip("'")
    
    if not os.path.exists(ref_path):
        print(f"错误：找不到参考图片 {ref_path}")
        return image
        
    ref_img = cv2.imdecode(np.fromfile(ref_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref_img is None:
        print("错误：无法读取参考图片")
        return image

    print(f"开始 SWD 风格迁移... Iterations={iter_count}, Projections={proj_count}")
    
    # 2. 预处理：转为 float32
    target = image.astype(np.float32)
    source = ref_img.astype(np.float32)
    
    h, w, c = target.shape
    
    # 展平像素 (N, 3)
    X = target.reshape(-1, 3)
    Y = source.reshape(-1, 3)
    
    iter_count = int(iter_count)
    proj_count = int(proj_count)
    
    # 3. 核心迭代
    for i in range(iter_count):
        # 生成随机投影向量并归一化 (3, proj_count)
        rand_mat = np.random.randn(3, proj_count).astype(np.float32)
        rand_mat /= np.sqrt(np.sum(rand_mat**2, axis=0, keepdims=True) + 1e-8)
        
        # 投影
        X_proj = np.dot(X, rand_mat)
        Y_proj = np.dot(Y, rand_mat)
        
        # 排序索引
        X_indices = np.argsort(X_proj, axis=0)
        
        # 对 Y 投影排序
        Y_proj_sorted = np.sort(Y_proj, axis=0)
        
        # 重采样 Y 以匹配 X 的像素数量 (线性插值)
        # 解决两张图尺寸不一致问题
        Y_resampled = np.zeros_like(X_proj)
        x_space = np.linspace(0, Y_proj.shape[0]-1, X_proj.shape[0])
        y_space = np.arange(Y_proj.shape[0])
        
        for p in range(proj_count):
            Y_resampled[:, p] = np.interp(x_space, y_space, Y_proj_sorted[:, p])
            
        # 计算当前 X 的排序投影
        X_proj_sorted = np.sort(X_proj, axis=0)
        
        # 计算差异
        diff = Y_resampled - X_proj_sorted
        
        # 将差异按原 X 的顺序重新排列
        diff_reordered = np.zeros_like(diff)
        for p in range(proj_count):
            diff_reordered[X_indices[:, p], p] = diff[:, p]
            
        # 反投影更新 X
        X_update = np.dot(diff_reordered, rand_mat.T)
        
        # 均值更新 (带简单的学习率调节)
        X += X_update / (proj_count / 3.0) # 这里的除数系数可作为平滑项
        
    print("SWD 处理完成")
    
    # 4. 重组并返回
    res_img = X.reshape(h, w, c)
    return np.clip(res_img, 0, 255).astype(np.uint8)

# =============================================================================
# 快速色彩迁移 (Reinhard算法)
# =============================================================================
def color_transfer_reinhard(image, ref_path, **kwargs):
    """
    Reinhard 极速色彩迁移 (基于统计学)
    比 SWD 快非常多，适合实时处理
    """
    # 1. 检查参考图
    if not ref_path: return image
    ref_path = str(ref_path).strip('"').strip("'")
    if not os.path.exists(ref_path):
        print(f"错误：找不到参考图片 {ref_path}")
        return image
        
    ref_img = cv2.imdecode(np.fromfile(ref_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref_img is None: return image

    # 2. 转换到 LAB 空间 (浮点数运算以保证精度)
    # LAB 空间将亮度(L)和色彩(A, B)分离，最适合做色彩迁移
    source = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 3. 计算均值和标准差
    # source: 原图, target: 参考风格图
    (l_mean_src, l_std_src) = cv2.meanStdDev(source)
    (l_mean_tar, l_std_tar) = cv2.meanStdDev(target)

    # 展平以便广播计算
    l_mean_src = l_mean_src.flatten()
    l_std_src = l_std_src.flatten()
    l_mean_tar = l_mean_tar.flatten()
    l_std_tar = l_std_tar.flatten()

    # 4. 执行色彩变换公式
    # 核心公式: Result = (Original - Mean_Src) * (Std_Tar / Std_Src) + Mean_Tar
    # 加上 1e-6 防止除以零
    result_lab = (source - l_mean_src) * (l_std_tar / (l_std_src + 1e-6)) + l_mean_tar

    # 5. 限制范围并转回 BGR
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    return result_bgr

# =============================================================================
# MKL 线性最优传输 (比 Reinhard 更精准，比 SWD 更快)
# =============================================================================
def color_transfer_mkl(image, ref_path, **kwargs):
    """
    Monge-Kantorovich Linear (MKL) 色彩迁移
    基于最优传输理论，考虑了颜色通道之间的相关性(协方差)，
    效果通常优于 Reinhard，且速度很快。
    """
    # 1. 检查参考图
    if not ref_path: return image
    ref_path = str(ref_path).strip('"').strip("'")
    if not os.path.exists(ref_path):
        print(f"错误：找不到参考图片 {ref_path}")
        return image
        
    ref_img = cv2.imdecode(np.fromfile(ref_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref_img is None: return image

    # 2. 预处理 (保持 RGB 空间即可，或者转 LAB 也可以，MKL 对色彩空间不敏感)
    # 为了计算协方差，我们需要 shape 为 (N, 3) 的 float32
    X = image.astype(np.float32) / 255.0
    Y = ref_img.astype(np.float32) / 255.0
    
    h, w, c = X.shape
    X = X.reshape(-1, 3)
    Y = Y.reshape(-1, 3)

    # 3. 计算均值和协方差矩阵
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)
    
    # rowvar=False 表示每一列是一个变量(R,G,B)
    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    # 4. 计算最优传输映射矩阵 T
    # 公式: T = A^(-1/2) * (A^(1/2) * B * A^(1/2))^(1/2) * A^(-1/2)
    # 其中 A = cov_x, B = cov_y
    # 为了计算矩阵平方根，使用 Eigendecomposition (特征分解)
    
    def matrix_sqrt(M):
        # 特征分解 M = V * D * V_inv
        eigval, eigvec = np.linalg.eigh(M)
        # 剔除极小的负特征值(数值误差)
        eigval = np.maximum(eigval, 0)
        # 构造 D^(1/2)
        D_sqrt = np.diag(np.sqrt(eigval))
        # 重组 M^(1/2) = V * D^(1/2) * V.T
        return eigvec @ D_sqrt @ eigvec.T

    def matrix_inv_sqrt(M):
        eigval, eigvec = np.linalg.eigh(M)
        eigval = np.maximum(eigval, 1e-6) # 防止除零
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigval))
        return eigvec @ D_inv_sqrt @ eigvec.T

    # 步骤拆解
    cov_x_sqrt = matrix_sqrt(cov_x)
    cov_x_inv_sqrt = matrix_inv_sqrt(cov_x)
    
    # 中间项: (cov_x_sqrt @ cov_y @ cov_x_sqrt)
    mid_term = cov_x_sqrt @ cov_y @ cov_x_sqrt
    mid_term_sqrt = matrix_sqrt(mid_term)
    
    # 最终变换矩阵 M
    M = cov_x_inv_sqrt @ mid_term_sqrt @ cov_x_inv_sqrt

    # 5. 应用变换
    # X_new = (X - mu_x) @ M + mu_y
    X_centered = X - mu_x
    X_transformed = X_centered @ M + mu_y

    # 6. 后处理
    result = X_transformed.reshape(h, w, c)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result
    return cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)

def generate_inverse_local_mask(image, rect, points_relative, tolerance=20, **kwargs):
    """
    局部掩码生成 (反向/排除):
    在选中区域内，将【不符合】用户选取点色彩范围的像素设为白色，符合的设为黑色。
    即：反向提取背景或异常点。
    """
    # 1. 复用原有逻辑计算正向掩码
    # 我们先调用现有的函数或者复用代码生成“符合颜色”的掩码
    # 为了保持独立性，建议直接复制 generate_local_mask 的代码并修改中间一步
    
    if rect is None or not points_relative:
        return image

    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    
    accumulated_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
    tol = float(tolerance)

    if not isinstance(points_relative, list):
        points_relative = [points_relative]

    for (px, py) in points_relative:
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
            
        seed_color = roi[py, px]
        
        lower_bound = np.clip(seed_color - tol, 0, 255).astype(np.uint8)
        upper_bound = np.clip(seed_color + tol, 0, 255).astype(np.uint8)
        
        mask_i = cv2.inRange(roi, lower_bound, upper_bound)
        accumulated_mask_roi = cv2.bitwise_or(accumulated_mask_roi, mask_i)
    
    # === 【修改核心】 ===
    # 此时 accumulated_mask_roi 中，符合颜色的点是白色的。
    # 我们需要取反：符合颜色变黑(0)，不符合变白(255)。
    inverse_mask_roi = cv2.bitwise_not(accumulated_mask_roi)
    # ==================
    
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # 将取反后的掩码放入全图
    full_mask[y:y+h, x:x+w] = inverse_mask_roi
    
    return cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)