import cv2
import numpy as np
import os
from datetime import datetime
from scipy.spatial import cKDTree  # 【新增】用于加速最近邻搜索
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
    # 准备输出目录
    time_str = datetime.now().strftime("%m%d_%H%M")
    output_dir = f"{time_str}_lap_step_out"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

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

    b_out = _process_ch_laplacian_advanced(b, 'Blue', get_params('b'), output_dir)
    g_out = _process_ch_laplacian_advanced(g, 'Green', get_params('g'), output_dir)
    r_out = _process_ch_laplacian_advanced(r, 'Red', get_params('r'), output_dir)
    
    merged = cv2.merge([b_out, g_out, r_out])
    res_uint8 = np.clip(merged * 255, 0, 255).astype(np.uint8)
    _save_step_image(res_uint8, output_dir, "Final_Result.jpg")
    
    print(f"过程文件已保存至: {output_dir}")
    return res_uint8

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