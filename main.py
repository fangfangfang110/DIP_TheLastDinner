import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os
from datetime import datetime

import image_methods  # 引入您的算法库

# =============================================================================
# 交互式 ROI (区域) 选择器 (保持不变)
# =============================================================================
class ROISelector(tk.Toplevel):
    def __init__(self, parent, cv_image, title="请框选目标区域 (按住鼠标拖拽 -> 确定)"):
        super().__init__(parent)
        self.title(title)
        self.cv_image = cv_image
        self.result_rect = None 
        
        screen_w = self.winfo_screenwidth() * 0.8
        screen_h = self.winfo_screenheight() * 0.8
        img_h, img_w = cv_image.shape[:2]
        
        self.scale = min(screen_w / img_w, screen_h / img_h, 1.0) 
        self.display_w = int(img_w * self.scale)
        self.display_h = int(img_h * self.scale)
        
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        self.canvas = tk.Canvas(self, width=self.display_w, height=self.display_h, cursor="cross")
        self.canvas.pack(side=tk.TOP)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        btn_frame = tk.Frame(self, pady=10, bg="#ddd")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(btn_frame, text="✅ 确定选区", command=self.on_confirm, width=15, bg="#90ee90", font=("bold", 10)).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="❌ 取消", command=self.on_cancel, width=10).pack(side=tk.RIGHT, padx=20)
        tk.Label(btn_frame, text="提示：按住鼠标左键在图中框选主体", bg="#ddd").pack(side=tk.LEFT)

        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        self.geometry(f"{self.display_w}x{self.display_h + 50}+{parent.winfo_rootx()+50}+{parent.winfo_rooty()+50}")
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=3)

    def on_mouse_drag(self, event):
        if self.start_x is None: return
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event): pass

    def on_confirm(self):
        if not self.rect_id:
            messagebox.showwarning("提示", "请先在图片上画一个框！")
            return
        coords = self.canvas.coords(self.rect_id)
        x1, y1, x2, y2 = coords
        xmin, ymin = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        
        real_x = int(xmin / self.scale)
        real_y = int(ymin / self.scale)
        real_w = int(width / self.scale)
        real_h = int(height / self.scale)
        
        max_h, max_w = self.cv_image.shape[:2]
        real_x = max(0, real_x)
        real_y = max(0, real_y)
        real_w = min(real_w, max_w - real_x)
        real_h = min(real_h, max_h - real_y)

        if real_w < 5 or real_h < 5:
            messagebox.showwarning("提示", "选区太小")
            return

        self.result_rect = (real_x, real_y, real_w, real_h)
        self.destroy()

    def on_cancel(self): self.destroy()


# =============================================================================
# 升级版：支持文件选择的多参数输入框
# =============================================================================
class MultiParamDialog(tk.Toplevel):
    def __init__(self, parent, title, param_configs, history_values=None):
        super().__init__(parent)
        self.title(title)
        self.result_data = None
        
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        frame = tk.Frame(canvas, padx=15, pady=15)
        
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0,0), window=frame, anchor="nw", tags="frame")
        
        def on_frame_configure(event): canvas.configure(scrollregion=canvas.bbox("all"))
        frame.bind("<Configure>", on_frame_configure)

        self.entries = {}
        self.param_types = {}

        tk.Label(frame, text="参数名称", font=("bold", 9)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Label(frame, text="输入值 / 路径", font=("bold", 9)).grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(frame, text="说明", font=("bold", 9), fg="#666").grid(row=0, column=2, sticky="w", padx=5)

        for i, cfg in enumerate(param_configs):
            row = i + 1
            key = cfg['key']
            p_type = cfg.get('type', 'number') 
            self.param_types[key] = p_type

            tk.Label(frame, text=cfg['label'] + ":").grid(row=row, column=0, sticky="e", padx=5, pady=3)
            
            initial_val = cfg['default']
            if history_values and key in history_values:
                initial_val = history_values[key]
            
            # 如果是文件类型，输入框加宽
            entry = tk.Entry(frame, width=30 if p_type == 'file' else 10)
            entry.insert(0, str(initial_val))
            entry.grid(row=row, column=1, padx=5, pady=3, sticky="w")
            self.entries[key] = entry
            
            # 【关键修改】添加文件浏览按钮
            if p_type == 'file':
                btn = tk.Button(frame, text="📂", width=3, command=lambda e=entry: self.browse_file(e))
                btn.grid(row=row, column=1, sticky="e", padx=5)

            tip = cfg.get('tip', '')
            tk.Label(frame, text=tip, fg="#555", font=("Arial", 8)).grid(row=row, column=2, sticky="w", padx=5)

        btn_frame = tk.Frame(self, pady=10)
        btn_frame.pack(side="bottom", fill="x")
        tk.Button(btn_frame, text="确定执行", command=self.on_ok, width=15, bg="#dddddd").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="取消", command=self.on_cancel, width=10).pack(side=tk.LEFT, padx=10)
        
        h = min(800, len(param_configs) * 40 + 100)
        self.geometry(f"600x{h}+{parent.winfo_rootx()+100}+{parent.winfo_rooty()+50}")
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def browse_file(self, entry_widget):
        # 允许选择所有常见的图片格式
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp *.tif")])
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def on_ok(self):
        try:
            data = {}
            for key, entry in self.entries.items():
                val_str = entry.get()
                p_type = self.param_types[key]
                if p_type == 'number':
                    data[key] = float(val_str)
                else:
                    data[key] = val_str # 字符串直接存
            self.result_data = data
            self.destroy()
        except ValueError:
            messagebox.showerror("错误", "数值参数必须为有效数字！")

    def on_cancel(self): self.destroy()

# =============================================================================
# 主应用程序
# =============================================================================
class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("数字图像处理专业平台 (IDW修复 + 智能日志)")
        self.root.geometry("1500x900") 
        
        self.cv_img_original = None
        self.cv_img_processed = None
        self.img_path_current = None
        
        # --- 日志系统重构 ---
        self.log_file_name = "image_processing_log.txt" 
        self.ui_log = []          # 列表1: 所有的尝试操作 (显示在界面上)
        self.persistent_log = []  # 列表2: 仅保存的成功操作 (写入文件)
        self.pending_log_entry = None # 暂存当前正在预览的算法信息
        
        self.param_history = {} 
        self.zoom_factor = 2.0
        self.zoom_focus = (0.5, 0.5) 

        self.methods_config = self._init_methods_config()
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_methods_config(self):
        def single(key, label, default, tip=""):
            return [{"key": key, "label": label, "default": default, "tip": tip}]

        return {
            "转灰度图": {"func": image_methods.to_gray, "params": [{"key": "dummy", "label": "无需参数", "default": 1}]},
            "GrabCut 交互式抠图": {
                "func": image_methods.grabcut_interactive,
                "interactive_roi": True,
                "params": [{"key": "iter_count", "label": "迭代次数", "default": 5}]
            },
            # --- 新增配置 ---
            "区域修补 (指定Mask图片)": {
                "func": image_methods.restoration_idw_external_mask,
                "interactive_roi": True,  # 开启鼠标选框
                "params": [
                    {
                        "key": "mask_path", 
                        "label": "二值图路径", 
                        "default": "", 
                        "type": "file",   # 指定类型为文件，会显示浏览按钮
                        "tip": "选择处理好的黑白二值图"
                    },
                    {
                        "key": "k_neighbors", "label": "参考点数量", "default": 5, "tip": "取周围最近的k个黑点"
                    }
                ]
            },
            "形态学边缘检测": {
                "func": image_methods.morph_edge_detection,
                "params": [
                    {"key": "kernel_size", "label": "线条粗细", "default": 3},
                    {"key": "mode", "label": "检测模式", "default": 0, "tip": "0=标准, 1=外边缘, 2=内边缘"}
                ]
            },
            "二值化处理 (黑白)": {
                "func": image_methods.binary_threshold,
                "params": [
                    {"key": "thresh_val", "label": "阈值(模式0)", "default": 127},
                    {"key": "method", "label": "算法模式", "default": 1, "tip": "0=手动, 1=Otsu, 2=自适应"}
                ]
            },
            "Gamma 亮度校正": {"func": image_methods.gamma_correction, "params": single("gamma", "Gamma值", 1.5, ">1 提亮")},
            "色彩饱和度": {"func": image_methods.color_saturation_boost, "params": single("scale", "倍数", 1.3, "1.0为原图")},
            "CLAHE 对比度增强": {"func": image_methods.clahe_enhance, "params": single("clip_limit", "Clip Limit", 3.0)},
            "形态学对比度": {"func": image_methods.morph_contrast_enhance, "params": single("kernel_size", "核大小", 15)},
            "USM 智能锐化": {"func": image_methods.unsharp_mask, "params": [{"key": "sigma", "label": "半径", "default": 2.0}, {"key": "amount", "label": "强度", "default": 1.5}]},
            "双边滤波降噪": {"func": image_methods.bilateral_filter_denoise, "params": [{"key": "d", "label": "直径", "default": 9}, {"key": "sigma", "label": "强度", "default": 75}]},
            "色度降噪": {"func": image_methods.chroma_denoise, "params": single("kernel_size", "核大小", 21)},
            "NLM 强力降噪": {"func": image_methods.nlm_denoise_colored, "params": [{"key": "h", "label": "亮度强度", "default": 10}, {"key": "h_color", "label": "色彩强度", "default": 10}, {"key": "templateWindowSize", "label": "模板大小", "default": 7}, {"key": "searchWindowSize", "label": "搜索大小", "default": 21}]},
            "暗通道去雾": {"func": image_methods.dehaze_dcp, "params": single("omega", "去雾程度", 0.95)},
            "暗通道去雾 (空间)": {"func": image_methods.dehaze_dcp_spatial, "params": [{"key": "omega_center", "label": "中心强度", "default": 0.98}, {"key": "omega_edge", "label": "边缘强度", "default": 0.60}, {"key": "radius", "label": "中心半径", "default": 0.60}]},
            "金字塔融合增强": {"func": image_methods.laplacian_pyramid_fusion, "params": [
                     {'key': 'b_base_h', 'label': '[蓝]去噪h', 'default': 9.0}, {'key': 'b_det_d', 'label': '[蓝]细节d', 'default': 5},
                     {'key': 'g_base_h', 'label': '[绿]去噪h', 'default': 5.0}, {'key': 'g_det_d', 'label': '[绿]细节d', 'default': 5},
                     {'key': 'r_base_h', 'label': '[红]去噪h', 'default': 5.0}, {'key': 'r_det_d', 'label': '[红]细节d', 'default': 5},
            ]}
        }

    def setup_ui(self):
        self.root.columnconfigure(1, weight=3)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 1. 左侧
        frame_left = tk.Frame(self.root, width=240, bg="#f0f0f0")
        frame_left.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        frame_left.pack_propagate(False) 
        
        tk.Label(frame_left, text="功能菜单", font=("微软雅黑", 12, "bold")).pack(pady=(10, 5))
        tk.Button(frame_left, text="📂 打开图片", command=self.load_image, bg="#add8e6", height=2).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(frame_left, text="💾 保存结果", command=self.save_image, bg="#90ee90", height=2).pack(fill=tk.X, padx=10, pady=5)
        
        algo_frame = tk.LabelFrame(frame_left, text="算法库", font=("bold", 10))
        algo_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(algo_frame)
        scrollbar = tk.Scrollbar(algo_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for name in self.methods_config:
            tk.Button(scrollable_frame, text=name, command=lambda n=name: self.apply_method(n), anchor="w").pack(fill=tk.X, pady=2)

        log_frame = tk.LabelFrame(frame_left, text="实时记录 (所有操作)", font=("bold", 9))
        log_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        self.log_text = tk.Text(log_frame, height=18, width=25, state=tk.DISABLED, font=("Consolas", 8), bg="#f5f5f5")
        self.log_text.pack(fill=tk.BOTH, padx=2, pady=2)

        # 2. 中间
        frame_center = tk.Frame(self.root, bg="#333")
        frame_center.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        frame_center.rowconfigure(0, weight=1)
        frame_center.rowconfigure(1, weight=1)
        frame_center.columnconfigure(0, weight=1)

        self.lbl_original = tk.Label(frame_center, text="[原图]", bg="#444", fg="#aaa")
        self.lbl_original.grid(row=0, column=0, sticky="nsew", pady=(0, 2))
        self.lbl_processed = tk.Label(frame_center, text="[结果]\n(在此处按住鼠标拖动查看细节)", bg="#444", fg="white", cursor="cross")
        self.lbl_processed.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        
        self.lbl_original.bind("<Configure>", lambda e: self.refresh_display())
        self.lbl_processed.bind("<ButtonPress-1>", self.on_click_zoom)
        self.lbl_processed.bind("<B1-Motion>", self.on_drag_zoom)

        # 3. 右侧
        frame_right = tk.Frame(self.root, width=320, bg="#e0e0e0")
        frame_right.grid(row=0, column=2, sticky="ns", padx=5, pady=5)
        frame_right.pack_propagate(False)

        tk.Label(frame_right, text="🔍 局部细节对比", font=("微软雅黑", 11, "bold")).pack(pady=10)
        detail_frame = tk.Frame(frame_right, bg="#e0e0e0")
        detail_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(detail_frame, text="Before", font=("Arial", 9)).pack(pady=(5, 0))
        self.lbl_zoom_ori = tk.Label(detail_frame, bg="#000", width=300, height=250)
        self.lbl_zoom_ori.pack(pady=2)
        
        tk.Label(detail_frame, text="After", font=("Arial", 9)).pack(pady=(10, 0))
        self.lbl_zoom_proc = tk.Label(detail_frame, bg="#000", width=300, height=250)
        self.lbl_zoom_proc.pack(pady=2)

        control_frame = tk.Frame(frame_right, pady=10)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(control_frame, text="放大倍率").pack()
        self.scale_zoom = tk.Scale(control_frame, from_=1.0, to=8.0, resolution=0.5, orient=tk.HORIZONTAL, command=self.update_zoom_view)
        self.scale_zoom.set(2.0)
        self.scale_zoom.pack(fill=tk.X, padx=20, pady=5)

    # =========================================================================
    # 逻辑部分
    # =========================================================================
    def log_operation(self, entry, is_ui_only=True):
        """
        :param entry: 日志内容
        :param is_ui_only: 如果为True，只显示在界面，不存入文件
        """
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_entry = f"{timestamp} {entry}"
        
        # 1. 更新 UI (始终执行)
        self.ui_log.append(full_entry)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, full_entry + "\n" + "-"*48 + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 2. 如果是文件保存操作，直接写入 persistent_log
        if not is_ui_only:
             self.persistent_log.append(full_entry)

    def save_session_log(self):
        """仅将 persistent_log (成功保存的操作) 写入文件"""
        if not self.persistent_log: return
        
        session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 头部信息
        content = f"\n{'='*60}\n会话保存时间: {session_start}\n初始文件: {self.img_path_current}\n注意: 仅包含最终保存的有效操作步骤\n{'='*60}\n" 
        content += "\n".join(self.persistent_log) + "\n\n"
        
        try:
            with open(self.log_file_name, 'a', encoding='utf-8') as f: f.write(content)
            print(f"有效日志已保存至 {self.log_file_name}")
        except Exception as e:
            print(f"日志保存失败: {e}")

    def on_closing(self):
        self.save_session_log()
        self.root.destroy()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp *.tif")])
        if not path: return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return
        self.cv_img_original = img
        self.cv_img_processed = img.copy() 
        self.img_path_current = path
        
        # 重置日志
        self.ui_log = [] 
        self.persistent_log = []
        self.pending_log_entry = None
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log_operation(f"文件加载成功: {os.path.basename(path)}")
        self.refresh_display()

    def apply_method(self, method_name):
        if self.cv_img_original is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
            
        config = self.methods_config[method_name]
        
        # 1. 交互式选框
        rect_roi = None
        if config.get("interactive_roi", False):
            selector = ROISelector(self.root, self.cv_img_original)
            if selector.result_rect is None:
                self.log_operation(f"❌ 取消操作: {method_name}")
                return 
            rect_roi = selector.result_rect
            self.log_operation(f"🖱️ 选区确定: {rect_roi}")
        
        # 2. 参数输入
        history = self.param_history.get(method_name, {})
        dialog = MultiParamDialog(self.root, f"参数: {method_name}", config["params"], history_values=history)
        if dialog.result_data is None: return 
        
        params = dialog.result_data
        
        # 记录尝试操作 (UI)
        ui_msg = f"🚀 正在执行: {method_name}\n   参数: {params}"
        self.log_operation(ui_msg)
        
        try:
            img_in = self.cv_img_original.copy()
            kwargs = params.copy()
            if rect_roi: kwargs['rect'] = rect_roi
                
            res = config["func"](img_in, **kwargs)
            self.cv_img_processed = res
            self.param_history[method_name] = params 
            
            self.refresh_display()
            self.log_operation(f"🎉 预览成功: {method_name}")
            
            # 【关键】将此操作暂存。只有用户点击保存，才写入文件日志
            self.pending_log_entry = f"应用算法: [{method_name}] | 参数: {params}"
            
        except Exception as e:
            messagebox.showerror("算法错误", str(e))
            self.log_operation(f"❌ 失败: {str(e)}")
            self.pending_log_entry = None

    def save_image(self):
        if self.cv_img_processed is None: return
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            # 1. 保存文件
            cv2.imencode(".jpg", self.cv_img_processed)[1].tofile(path)
            
            # 2. 更新原图状态
            self.cv_img_original = self.cv_img_processed.copy()
            
            # 3. 【核心日志逻辑】
            # 因为保存了，所以上一步的"暂存操作"变成了"永久历史"
            if self.pending_log_entry:
                self.log_operation(f"✅ 确认应用并保存: {self.pending_log_entry}", is_ui_only=False)
                self.pending_log_entry = None # 清空暂存
            
            # 记录保存动作本身
            save_msg = f"💾 文件已保存至: {os.path.basename(path)}"
            self.log_operation(save_msg, is_ui_only=False)
            
            self.refresh_display()

    # (以下显示相关函数保持不变)
    def refresh_display(self):
        if self.cv_img_original is None: return
        w, h = self.lbl_original.winfo_width(), self.lbl_original.winfo_height()
        if w < 10: return
        img_h, img_w = self.cv_img_original.shape[:2]
        scale = min(w / img_w, h / img_h)
        dw, dh = int(img_w * scale), int(img_h * scale)
        self._render_view(self.lbl_original, self.cv_img_original, dw, dh, scale, w, h)
        target = self.cv_img_processed if self.cv_img_processed is not None else self.cv_img_original
        self._render_view(self.lbl_processed, target, dw, dh, scale, w, h)
        self.update_zoom_view()

    def _render_view(self, label, cv_img, target_w, target_h, scale, container_w, container_h):
        if cv_img is None: return
        label.display_scale = scale
        label.display_offset_x = (container_w - target_w) // 2
        label.display_offset_y = (container_h - target_h) // 2
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((target_w, target_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        label.config(image=tk_img)
        label.image = tk_img

    def on_click_zoom(self, event): self._update_focus_from_mouse(event)
    def on_drag_zoom(self, event): self._update_focus_from_mouse(event)
    def _update_focus_from_mouse(self, event):
        if self.cv_img_processed is None: return
        lbl = self.lbl_processed
        if not hasattr(lbl, 'display_scale'): return
        mx = event.x - lbl.display_offset_x
        my = event.y - lbl.display_offset_y
        real_w = self.cv_img_processed.shape[1] * lbl.display_scale
        real_h = self.cv_img_processed.shape[0] * lbl.display_scale
        self.zoom_focus = (np.clip(mx / real_w, 0, 1), np.clip(my / real_h, 0, 1))
        self.update_zoom_view()

    def update_zoom_view(self, _=None):
        if self.cv_img_processed is None or self.cv_img_original is None: return
        view_w, view_h = 300, 250 
        scale = float(self.scale_zoom.get())
        crop_w, crop_h = int(view_w / scale), int(view_h / scale)
        img_h, img_w = self.cv_img_processed.shape[:2]
        cx, cy = int(self.zoom_focus[0] * img_w), int(self.zoom_focus[1] * img_h)
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2, y2 = min(img_w, x1 + crop_w), min(img_h, y1 + crop_h)
        if x2 - x1 < crop_w: x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h: y1 = max(0, y2 - crop_h)
        
        def show_crop(img_source, lbl_target):
            if img_source.shape[:2] != (img_h, img_w):
                sh, sw = img_source.shape[:2]
                sx1, sx2 = int(x1 * (sw/img_w)), int(x2 * (sw/img_w))
                sy1, sy2 = int(y1 * (sh/img_h)), int(y2 * (sh/img_h))
                crop = img_source[sy1:sy2, sx1:sx2]
            else: crop = img_source[y1:y2, x1:x2]
            if crop.size == 0: return
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((view_w, view_h), Image.Resampling.NEAREST)
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl_target.config(image=tk_img)
            lbl_target.image = tk_img
        show_crop(self.cv_img_original, self.lbl_zoom_ori)
        show_crop(self.cv_img_processed, self.lbl_zoom_proc)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()