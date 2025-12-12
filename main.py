import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import threading  # <--- 【新增】导入线程模块
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os
from datetime import datetime
import image_methods

# =============================================================================
# 交互式 ROI (区域) 选择器
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
# 四点透视选择器
# =============================================================================
class PointSelector(tk.Toplevel):
    def __init__(self, parent, cv_image, title="请依次点击四个角 (左上->右上->右下->左下)"):
        super().__init__(parent)
        self.title(title)
        self.cv_image = cv_image
        self.result_points = None 
        self.selected_points = [] 
        
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
        
        self.lbl_status = tk.Label(btn_frame, text="当前进度: 0/4", font=("bold", 10), bg="#ddd", fg="blue")
        self.lbl_status.pack(side=tk.LEFT, padx=20)
        
        tk.Button(btn_frame, text="❌ 撤销上一点", command=self.undo_point, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="✅ 确认变换", command=self.on_confirm, width=15, bg="#90ee90", font=("bold", 10)).pack(side=tk.RIGHT, padx=20)
        tk.Button(btn_frame, text="取消", command=self.on_cancel, width=10).pack(side=tk.RIGHT, padx=5)

        self.canvas.bind("<ButtonPress-1>", self.on_click)
        
        self.geometry(f"{self.display_w}x{self.display_h + 50}+{parent.winfo_rootx()+50}+{parent.winfo_rooty()+50}")
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def on_click(self, event):
        if len(self.selected_points) >= 4: return
        x, y = event.x, event.y
        self.selected_points.append((x, y))
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="white", tags=f"p{len(self.selected_points)}")
        self.canvas.create_text(x, y-15, text=str(len(self.selected_points)), fill="yellow", font=("bold", 12), tags=f"t{len(self.selected_points)}")
        if len(self.selected_points) > 1:
            prev = self.selected_points[-2]
            curr = self.selected_points[-1]
            self.canvas.create_line(prev[0], prev[1], curr[0], curr[1], fill="red", width=2, tags=f"l{len(self.selected_points)}")
        if len(self.selected_points) == 4:
            p4, p1 = self.selected_points[-1], self.selected_points[0]
            self.canvas.create_line(p4[0], p4[1], p1[0], p1[1], fill="red", width=2, tags="l_close")
        self.update_status()

    def undo_point(self):
        if not self.selected_points: return
        n = len(self.selected_points)
        self.selected_points.pop()
        self.canvas.delete(f"p{n}")
        self.canvas.delete(f"t{n}")
        self.canvas.delete(f"l{n}")
        self.canvas.delete("l_close")
        self.update_status()

    def update_status(self):
        self.lbl_status.config(text=f"当前进度: {len(self.selected_points)}/4")

    def on_confirm(self):
        if len(self.selected_points) != 4:
            messagebox.showwarning("提示", "请准确选取 4 个角点！")
            return
        real_points = []
        for (sx, sy) in self.selected_points:
            rx = int(sx / self.scale)
            ry = int(sy / self.scale)
            real_points.append([rx, ry])
        self.result_points = np.array(real_points, dtype=np.float32)
        self.destroy()

    def on_cancel(self): self.destroy()


# =============================================================================
# 单点像素选择器 (用于在 ROI 中取色)
# =============================================================================
class PixelSelector(tk.Toplevel):
    def __init__(self, parent, cv_roi, title="请点击目标颜色点 (支持多点)"):
        super().__init__(parent)
        self.title(title)
        self.cv_roi = cv_roi
        self.result_points = []  # 【修改】存储列表 [(x1,y1), (x2,y2), ...]
        self.selected_points = [] # 内部临时存储
        
        # 放大显示逻辑保持不变
        roi_h, roi_w = cv_roi.shape[:2]
        # 计算适合屏幕的缩放比例
        scale_w = 800 / roi_w
        scale_h = 800 / roi_h
        # self.scale = min(scale_w, scale_h, 10.0) 
        
        # 【修改】允许缩放比例小于 1，以便大图能缩小显示
        if scale_w < 1 or scale_h < 1:
            self.scale = min(scale_w, scale_h)
        else:
            self.scale = min(scale_w, scale_h, 10.0)
        self.display_w = int(roi_w * self.scale)
        self.display_h = int(roi_h * self.scale)
        
        rgb = cv2.cvtColor(cv_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((self.display_w, self.display_h), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        self.canvas = tk.Canvas(self, width=self.display_w, height=self.display_h, cursor="tcross")
        self.canvas.pack(side=tk.TOP)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        # 【新增】底部按钮区
        btn_frame = tk.Frame(self, pady=10, bg="#ddd")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(btn_frame, text="左键取点，右键撤销", bg="#ddd", fg="#555").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="✅ 完成选择", command=self.on_confirm, bg="#90ee90", font=("bold", 10), width=15).pack(side=tk.RIGHT, padx=10)
        tk.Button(btn_frame, text="❌ 清空", command=self.on_clear, width=10).pack(side=tk.RIGHT, padx=5)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.undo_last) # 右键撤销
        
        self.geometry(f"{self.display_w}x{self.display_h + 50}+{parent.winfo_rootx()+100}+{parent.winfo_rooty()+100}")
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def on_click(self, event):
        # 映射坐标
        real_x = int(event.x / self.scale)
        real_y = int(event.y / self.scale)
        
        h, w = self.cv_roi.shape[:2]
        real_x = np.clip(real_x, 0, w-1)
        real_y = np.clip(real_y, 0, h-1)
        
        # 记录点并绘制标记
        self.selected_points.append((real_x, real_y))
        
        # 在界面上画个圈标记
        r = 4
        tag_id = f"pt_{len(self.selected_points)}"
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="red", outline="white", width=2, tags=tag_id)

    def undo_last(self, event=None):
        if not self.selected_points: return
        self.selected_points.pop()
        # 删除最后一个标记
        self.canvas.delete(f"pt_{len(self.selected_points) + 1}")

    def on_clear(self):
        self.selected_points = []
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_confirm(self):
        if not self.selected_points:
            messagebox.showwarning("提示", "请至少选择一个点")
            return
        self.result_points = self.selected_points
        self.destroy()

# =============================================================================
# 手动坐标输入框
# =============================================================================
class ManualCoordDialog(tk.Toplevel):
    def __init__(self, parent, max_w, max_h):
        super().__init__(parent)
        self.title("输入区域坐标")
        self.result = None
        self.max_w, self.max_h = max_w, max_h
        
        tk.Label(self, text=f"图片尺寸: {max_w} x {max_h}", fg="#666").pack(pady=5)
        frame = tk.Frame(self, padx=20, pady=10)
        frame.pack()
        
        self.entries = {}
        for i, (lbl, key) in enumerate(zip(["X (起点)", "Y (起点)", "W (宽度)", "H (高度)"], ['x', 'y', 'w', 'h'])):
            tk.Label(frame, text=lbl).grid(row=i, column=0, sticky="e")
            ent = tk.Entry(frame, width=10)
            ent.grid(row=i, column=1, padx=5, pady=2)
            self.entries[key] = ent
            
        btn_frame = tk.Frame(self, pady=10)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="确定", command=self.on_confirm, bg="#90ee90").pack(side=tk.LEFT, padx=40)
        tk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=40)
        
        self.geometry(f"300x220+{parent.winfo_rootx()+100}+{parent.winfo_rooty()+100}")
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def on_confirm(self):
        try:
            x, y = int(self.entries['x'].get()), int(self.entries['y'].get())
            w, h = int(self.entries['w'].get()), int(self.entries['h'].get())
            if x<0 or y<0 or w<=0 or h<=0 or x+w>self.max_w or y+h>self.max_h:
                messagebox.showwarning("错误", "坐标越界或无效")
                return
            self.result = (x, y, w, h)
            self.destroy()
        except: messagebox.showerror("错误", "请输入整数")

# =============================================================================
# 色彩统计选择器
# =============================================================================
class ColorStatSelector(tk.Toplevel):
    def __init__(self, parent, cv_image, rect):
        super().__init__(parent)
        self.title("请勾选目标主色调")
        self.result_colors = None
        
        frame_main = tk.Frame(self, padx=10, pady=10)
        frame_main.pack(fill=tk.BOTH, expand=True)
        
        # 预览区
        x, y, w, h = rect
        roi = cv_image[y:y+h, x:x+w]
        scale = 150 / max(h, 1)
        dw = max(50, int(w * scale))
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).resize((dw, 150), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        tk.Label(frame_main, text="区域预览").pack()
        tk.Label(frame_main, image=self.tk_img, bg="black").pack(pady=5)
        
        # 列表区
        canvas = tk.Canvas(frame_main, height=200)
        scroll = tk.Scrollbar(frame_main, command=canvas.yview)
        list_frame = tk.Frame(canvas)
        list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats = image_methods.get_dominant_colors_kmeans(cv_image, rect, k=12)
        self.vars, self.colors = [], []
        
        for i, (bgr, count) in enumerate(self.stats):
            row = tk.Frame(list_frame)
            row.pack(fill=tk.X, pady=2)
            hex_c = '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])
            tk.Label(row, bg=hex_c, width=4, relief="solid").pack(side=tk.LEFT, padx=5)
            var = tk.IntVar(value=1 if i==0 else 0)
            tk.Checkbutton(row, text=f"占比: {count} px", variable=var).pack(side=tk.LEFT)
            self.vars.append(var)
            self.colors.append(bgr)

        tk.Button(self, text="确定", command=self.on_confirm, bg="#90ee90", width=20).pack(pady=10)
        self.geometry(f"400x500+{parent.winfo_rootx()+100}+{parent.winfo_rooty()+100}")
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def on_confirm(self):
        selected = [self.colors[i] for i, v in enumerate(self.vars) if v.get() == 1]
        if not selected:
            messagebox.showwarning("提示", "请至少选一种颜色")
            return
        self.result_colors = selected
        self.destroy()

# =============================================================================
# 自适应多参数输入框
# =============================================================================
class MultiParamDialog(tk.Toplevel):
    def __init__(self, parent, title, param_configs, history_values=None):
        super().__init__(parent)
        self.title(title)
        self.result_data = None
        
        # 1. 基础布局配置
        self.minsize(600, 300) # 设置最小尺寸
        
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        frame = tk.Frame(canvas, padx=15, pady=15)
        
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # 关键修改1: frame 宽度绑定到 canvas 宽度，实现自适应拉伸
        canvas_window = canvas.create_window((0,0), window=frame, anchor="nw", tags="frame")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        frame.bind("<Configure>", on_frame_configure)

        # 关键修改2: 配置列权重，让说明列自动占用剩余空间
        frame.columnconfigure(0, weight=0) # 参数名: 固定
        frame.columnconfigure(1, weight=1) # 输入框: 稍微拉伸
        frame.columnconfigure(2, weight=2) # 说明: 主要拉伸区

        self.entries = {}
        self.param_types = {}

        # 表头
        tk.Label(frame, text="参数名称", font=("bold", 9)).grid(row=0, column=0, sticky="w", padx=5)
        tk.Label(frame, text="输入值 / 路径", font=("bold", 9)).grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(frame, text="说明", font=("bold", 9), fg="#666").grid(row=0, column=2, sticky="w", padx=5)

        for i, cfg in enumerate(param_configs):
            row = i + 1
            key = cfg['key']
            p_type = cfg.get('type', 'number') 
            self.param_types[key] = p_type

            # Label
            tk.Label(frame, text=cfg['label'] + ":", anchor="e").grid(row=row, column=0, sticky="e", padx=5, pady=3)
            
            initial_val = cfg['default']
            if history_values and key in history_values:
                initial_val = history_values[key]
            
            # Entry & Button
            if p_type == 'file':
                # 关键修改3: 对于文件输入，使用 Frame 组合，防止按钮和输入框重叠
                f_container = tk.Frame(frame)
                f_container.grid(row=row, column=1, sticky="ew", padx=5)
                
                entry = tk.Entry(f_container)
                entry.insert(0, str(initial_val))
                entry.pack(side="left", fill="x", expand=True)
                self.entries[key] = entry
                
                btn = tk.Button(f_container, text="📂", command=lambda e=entry: self.browse_file(e))
                btn.pack(side="right", padx=(5,0))
            else:
                # 普通数字输入
                entry = tk.Entry(frame)
                entry.insert(0, str(initial_val))
                # 数字输入框不需要太宽，但可以设置sticky="w"让它靠左
                entry.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
                self.entries[key] = entry

            # Tip (说明)
            tip = cfg.get('tip', '')
            # 关键修改4: 增加 wraplength 实现自动换行，防止文字被截断
            tk.Label(frame, text=tip, fg="#555", font=("Arial", 8), 
                     wraplength=380, justify="left").grid(row=row, column=2, sticky="w", padx=5)

        # 按钮区
        btn_frame = tk.Frame(self, pady=10)
        btn_frame.pack(side="bottom", fill="x")
        tk.Button(btn_frame, text="确定执行", command=self.on_ok, width=15, bg="#dddddd").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="取消", command=self.on_cancel, width=10).pack(side=tk.LEFT, padx=10)
        
        # 初始高度计算
        h = min(900, len(param_configs) * 70 + 100)
        self.geometry(f"900x{h}+{parent.winfo_rootx()+50}+{parent.winfo_rooty()+50}")
        
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def browse_file(self, entry_widget):
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
                    data[key] = val_str 
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
        self.root.title("数字图像处理专业平台")
        self.root.geometry("1500x900") 
        
        self.cv_img_original = None
        self.cv_img_processed = None
        self.img_path_current = None
        
        self.log_file_name = "image_processing_log.txt" 
        self.ui_log = []          
        self.persistent_log = []  
        self.pending_log_entry = None 
        
        self.param_history = {} 
        self.zoom_factor = 2.0
        self.zoom_focus = (0.5, 0.5) 

        self.methods_config = self._init_methods_config()
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_methods_config(self):
        # 辅助函数：快速生成单参数配置
        def single(key, label, default, tip=""):
            return [{"key": key, "label": label, "default": default, "tip": tip}]

        return {
           "画布扩展 (通用版)": {
                "func": image_methods.canvas_expand_universal,
                "params": [
                    {"key": "pad_top", "label": "上边距", "default": 200},
                    {"key": "pad_bottom", "label": "下边距", "default": 200},
                    {"key": "pad_left", "label": "左边距", "default": 200},
                    {"key": "pad_right", "label": "右边距", "default": 200},
                    {"key": "algo_mode", "label": "扩展算法", "default": 2, "tip": "0=纯黑, 1=镜像, 2=复制边缘(配合优化), 3=流体, 4=Telea"},
                    
                    # 【修改】更新提示文字
                    {"key": "clean_strength", "label": "边缘清洗/阈值", "default": 100, "tip": "0=关闭。输入<20为滤波强度；输入>20为黑点阈值(推荐100)，可强力去除大块黑斑。"},
                    
                    {"key": "radius", "label": "修补半径", "default": 3.0, "tip": "仅对算法3、4有效"}
                ]
            },
            # 注意：这里去掉了 params，意味着不需要弹出参数框
            "画布裁剪 (ROI选取)": {
                "func": image_methods.canvas_crop,
                "interactive_roi": True, 
                "params": [] 
            },
            "画布裁剪 (指定边距)": {
                "func": image_methods.canvas_crop_margin,
                "params": [
                    {"key": "crop_top", "label": "顶部裁剪像素", "default": 0},
                    {"key": "crop_bottom", "label": "底部裁剪像素", "default": 0},
                    {"key": "crop_left", "label": "左侧裁剪像素", "default": 0},
                    {"key": "crop_right", "label": "右侧裁剪像素", "default": 0}
                ]
            },
            "交互式透视变换校正": {
                "func": image_methods.perspective_correction,
                "interactive_points": True,
                "params": [
                    {"key": "target_width", "label": "输出宽度(0自动)", "default": 0, "tip": "若0则自动计算"},
                    {"key": "target_height", "label": "输出高度(0自动)", "default": 0, "tip": "例如1000"}
                ]
            },
            "转灰度图": {
                "func": image_methods.to_gray,
                "params": [{"key": "dummy", "label": "无需参数", "default": 1}]
            },
            "GrabCut 交互式抠图": {
                "func": image_methods.grabcut_interactive,
                "interactive_roi": True,
                "params": [{"key": "iter_count", "label": "迭代次数", "default": 5}]
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
           "局部掩码生成 (颜色阈值)": {
                "func": image_methods.generate_local_mask_by_colors,
                "roi_and_color_stat": True,  # 使用新交互逻辑
                "params": [
                    {"key": "tolerance", "label": "容差范围", "default": 20, "tip": "颜色浮动范围"},
                    {"key": "inverse", "label": "模式", "default": 0, "tip": "无需修改 (0=正向)"} # 隐藏参数，默认0
                ]
            },
            "局部掩码生成 (反向排除)": {
                "func": image_methods.generate_local_mask_by_colors,
                "roi_and_color_stat": True,  # 使用新交互逻辑
                "params": [
                    {"key": "tolerance", "label": "排除容差", "default": 20, "tip": "剔除颜色的范围"},
                    {"key": "inverse", "label": "模式", "default": 1, "tip": "无需修改 (1=反向)"} # 隐藏参数，默认1
                ]
            },
            "区域修补 (指定Mask图片)": {
                "func": image_methods.restoration_idw_external_mask,
                "interactive_roi": True,
                "params": [
                    {"key": "mask_path", "label": "二值图路径", "default": "", "type": "file", "tip": "选择处理好的黑白二值图"},
                    {"key": "k_neighbors", "label": "参考点数量", "default": 5, "tip": "取周围最近的k个黑点"}
                ]
            },
            "Gamma 亮度校正": {
                "func": image_methods.gamma_correction,
                "params": single("gamma", "Gamma值", 1.5, ">1 提亮")
            },
            "Laplacian特征融合": {
                "func": image_methods.laplacian_pyramid_fusion,
                "params": [
                    {'key': 'b_base_h',      'label': '[蓝]底图-去噪强度h', 'default': 9.0},
                    {'key': 'b_base_temp',   'label': '[蓝]底图-模板大小',   'default': 7, 'tip': '奇数'},
                    {'key': 'b_base_search', 'label': '[蓝]底图-搜索大小',   'default': 21, 'tip': '奇数'},
                    {'key': 'b_det_d',       'label': '[蓝]细节-双边直径d', 'default': 5, 'tip': '0为不处理'},
                    {'key': 'b_det_sc',      'label': '[蓝]细节-双边色差Sigma', 'default': 30, 'tip': '0为不处理'},
                    {'key': 'b_det_ss',      'label': '[蓝]细节-双边空间Sigma', 'default': 30, 'tip': '0为不处理'},
                    {'key': 'b_w_l0',        'label': '[蓝]L0层权重',   'default': 0.2, 'tip': '噪点多则调小'},
                    {'key': 'b_w_l1',        'label': '[蓝]L1层权重',   'default': 0.9},
                    
                    {'key': 'g_base_h',      'label': '[绿]底图-去噪强度h', 'default': 5.0},
                    {'key': 'g_base_temp',   'label': '[绿]底图-模板大小',   'default': 7},
                    {'key': 'g_base_search', 'label': '[绿]底图-搜索大小',   'default': 21},
                    {'key': 'g_det_d',       'label': '[绿]细节-双边直径d', 'default': 5},
                    {'key': 'g_det_sc',      'label': '[绿]细节-双边色差Sigma', 'default': 30},
                    {'key': 'g_det_ss',      'label': '[绿]细节-双边空间Sigma', 'default': 30},
                    {'key': 'g_w_l0',        'label': '[绿]L0层权重',   'default': 0.3},
                    {'key': 'g_w_l1',        'label': '[绿]L1层权重',   'default': 1.2},
                    
                    {'key': 'r_base_h',      'label': '[红]底图-去噪强度h', 'default': 5.0},
                    {'key': 'r_base_temp',   'label': '[红]底图-模板大小',   'default': 7},
                    {'key': 'r_base_search', 'label': '[红]底图-搜索大小',   'default': 21},
                    {'key': 'r_det_d',       'label': '[红]细节-双边直径d', 'default': 5},
                    {'key': 'r_det_sc',      'label': '[红]细节-双边色差Sigma', 'default': 30},
                    {'key': 'r_det_ss',      'label': '[红]细节-双边空间Sigma', 'default': 30},
                    {'key': 'r_w_l0',        'label': '[红]L0层权重',   'default': 0.3},
                    {'key': 'r_w_l1',        'label': '[红]L1层权重',   'default': 1.2},
                ]
            },
            "Laplacian: 提取特征": {
                "func": image_methods.laplacian_extract_save,
                "params": [
                    {"key": "levels", "label": "提取层数", "default": 3, "tip": "层数越多，分离出的低频信息越少"}
                ]
            },
            "Laplacian: 注入特征(恢复细节)": {
                "func": image_methods.laplacian_inject_layer,
                "params": [
                    {"key": "layer_path", "label": "特征层文件", "default": "", "type": "file", "tip": "选择 process 文件夹下保存的 Level_x.png"},
                    {"key": "strength", "label": "注入强度", "default": 1.0, "tip": "1.0=还原, >1=锐化, 负数=模糊"}
                ]
            },
            "拉普拉斯: 多层自定义融合": {
                "func": image_methods.laplacian_inject_multilevel,
                "params": [
                    {"key": "anchor_path", "label": "特征定位", "default": "", "type": "file", "tip": "进入特征文件夹，选中任意一个Level文件即可"},
                    {"key": "weights_str", "label": "各层权重", "default": "1.0, 0.5, 0.2", "tip": "逗号分隔，分别对应第0层、第1层、第2层..."}
                ]
            },
            
            "Reinhard 快速色彩迁移": {
                "func": image_methods.color_transfer_reinhard,
                "params": [
                    {"key": "ref_path", "label": "参考图路径", "default": "", "type": "file", "tip": "瞬间完成色彩风格模仿"}
                ]
            },
            "SWD 风格迁移 (色彩同化)": {
                "func": image_methods.color_transfer_swd,
                "params": [
                    {"key": "ref_path", "label": "参考图路径", "default": "", "type": "file", "tip": "选择一张你想模仿其色彩风格的图片"},
                    {"key": "iter_count", "label": "迭代次数", "default": 30, "tip": "越大效果越明显但越慢"},
                    {"key": "proj_count", "label": "投影数量", "default": 64, "tip": "建议 50-100"}
                ]
            },
                "MKL 线性最优传输 (推荐)": {
                "func": image_methods.color_transfer_mkl,
                "params": [
                    {"key": "ref_path", "label": "参考图路径", "default": "", "type": "file", "tip": "速度快且色彩还原度高"}
                ]
            },
            "色彩饱和度": {
                "func": image_methods.color_saturation_boost,
                "params": single("scale", "倍数", 1.3, "1.0为原图")
            },
            "CLAHE 对比度增强": {
                "func": image_methods.clahe_enhance,
                "params": single("clip_limit", "Clip Limit", 3.0)
            },
            "形态学对比度": {
                "func": image_methods.morph_contrast_enhance,
                "params": single("kernel_size", "核大小", 15)
            },
            "USM 智能锐化": {
                "func": image_methods.unsharp_mask,
                "params": [
                    {"key": "sigma", "label": "半径", "default": 2.0},
                    {"key": "amount", "label": "强度", "default": 1.5}
                ]
            },
            "双边滤波降噪": {
                "func": image_methods.bilateral_filter_denoise,
                "params": [
                    {"key": "d", "label": "直径", "default": 9},
                    {"key": "sigma", "label": "强度", "default": 75}
                ]
            },
            "色度降噪": {
                "func": image_methods.chroma_denoise,
                "params": single("kernel_size", "核大小", 21)
            },
            "NLM 强力降噪": {
                "func": image_methods.nlm_denoise_colored,
                "params": [
                    {"key": "h", "label": "亮度强度", "default": 10},
                    {"key": "h_color", "label": "色彩强度", "default": 10},
                    {"key": "templateWindowSize", "label": "模板大小", "default": 7},
                    {"key": "searchWindowSize", "label": "搜索大小", "default": 21}
                ]
            },
            "暗通道去雾": {
                "func": image_methods.dehaze_dcp,
                "params": single("omega", "去雾程度", 0.95)
            },
            "暗通道去雾 (局部)": {
                "func": image_methods.dehaze_dcp_spatial,
                "params": [
                    {"key": "omega_center", "label": "中心强度", "default": 0.7},
                    {"key": "omega_edge", "label": "边缘强度", "default": 0.30},
                    {"key": "radius", "label": "中心半径", "default": 0.60}
                ]
            }
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
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", tags="frame")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            # Windows/MacOS 滚轮处理
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            # 鼠标进入区域：绑定滚轮事件 (使用 bind_all 捕获全局滚轮)
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            # Linux 系统兼容 (Button-4 上滚, Button-5 下滚)
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        def _unbind_mousewheel(event):
            # 鼠标离开区域：解绑，防止干扰其他组件(如日志栏)
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        # 将进入/离开事件绑定到最外层的容器 algo_frame 上
        algo_frame.bind('<Enter>', _bind_mousewheel)
        algo_frame.bind('<Leave>', _unbind_mousewheel)

        for name in self.methods_config:
            tk.Button(scrollable_frame, text=name, command=lambda n=name: self.apply_method(n), anchor="w").pack(fill=tk.X, pady=2)

        log_frame = tk.LabelFrame(frame_left, text="实时操作记录", font=("bold", 9))
        log_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        # === 【新增】进度条组件 ===
        # mode='indeterminate' 表示不确定进度（来回滚动），适合 OpenCV 这种无法预知剩余时间的操作
        self.progress_bar = ttk.Progressbar(log_frame, mode='indeterminate', length=200)
        self.progress_bar.pack(fill=tk.X, padx=2, pady=(0, 5), side=tk.BOTTOM)
        # ========================
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
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full_entry = f"{timestamp} {entry}"
        
        # 1. 更新 UI 界面
        self.ui_log.append(full_entry)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, full_entry + "\n" + "-"*48 + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 2. 如果是重要操作(is_ui_only=False)，立即写入文件
        if not is_ui_only:
             self.persistent_log.append(full_entry)
             try:
                 # 使用追加模式 'a' 即时写入
                 with open(self.log_file_name, 'a', encoding='utf-8') as f:
                     # 如果是本次启动的第一条，加个分割线和时间头
                     if len(self.persistent_log) == 1:
                         session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                         header = f"\n{'='*60}\n会话记录: {session_start} | 文件: {os.path.basename(self.img_path_current)}\n{'='*60}\n"
                         f.write(header)
                     
                     f.write(full_entry + "\n")
             except Exception as e:
                 print(f"日志写入失败: {e}")

    def save_session_log(self):
        if not self.persistent_log: return
        session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"\n{'='*60}\n会话保存时间: {session_start}\n初始文件: {self.img_path_current}\n注意: 仅包含最终保存的有效操作步骤\n{'='*60}\n" 
        content += "\n".join(self.persistent_log) + "\n\n"
        try:
            with open(self.log_file_name, 'a', encoding='utf-8') as f: f.write(content)
            print(f"有效日志已保存至 {self.log_file_name}")
        except Exception as e:
            print(f"日志保存失败: {e}")

    def on_closing(self):
        #self.save_session_log()
        self.root.destroy()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp *.tif")])
        if not path: return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return
        self.cv_img_original = img
        self.cv_img_processed = img.copy() 
        self.img_path_current = path
        
        self.ui_log = [] 
        self.persistent_log = []
        self.pending_log_entry = None
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log_operation(f"文件加载成功: {os.path.basename(path)}")
        self.refresh_display()

    def apply_method(self, method_name):
        # 1. 基础检查（保留）
        if self.cv_img_original is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
            
        config = self.methods_config[method_name]

        # =========================================================
        # 第一阶段：UI 交互部分
        # (这部分代码完全照搬原来的，负责获取用户输入)
        # =========================================================
        rect_roi = None
        points = None
        points_relative = None # 这是一个列表
        target_colors = None

        # === 处理：区域选择 (手动/鼠标) + 色彩统计 ===
        # 替代了旧的 roi_and_point 逻辑
        if config.get("roi_and_color_stat", False):
            # 1. 询问区域选择方式
            choice = messagebox.askyesno(
                "第一步：选择区域", 
                "请选择区域指定方式：\n\n【是 (Yes)】 手动输入坐标 (X,Y,W,H)\n【否 (No)】  鼠标框选"
            )
            
            if choice: # 手动输入
                h, w = self.cv_img_original.shape[:2]
                dlg = ManualCoordDialog(self.root, w, h)
                if dlg.result is None: return
                rect_roi = dlg.result
                self.log_operation(f"⌨️ 手动坐标: {rect_roi}")
            else: # 鼠标框选
                selector = ROISelector(self.root, self.cv_img_original, title="请框选分析区域")
                if selector.result_rect is None: return
                rect_roi = selector.result_rect
                self.log_operation(f"🖱️ 鼠标框选: {rect_roi}")
            
            # 2. 弹出色彩统计选择
            # 将确定好的 rect_roi 传入进行分析
            selector_color = ColorStatSelector(self.root, self.cv_img_original, rect_roi)
            if not selector_color.result_colors: return
            
            # 3. 准备参数
            target_colors = selector_color.result_colors  # 赋值给第一步定义的变量
            # kwargs['target_colors'] = selector_color.result_colors
            # kwargs['rect'] = rect_roi
            
            self.log_operation(f"🎨 选定主色: {len(selector_color.result_colors)} 种")

        # 情况B: 仅 ROI
        elif config.get("interactive_roi", False):
            selector = ROISelector(self.root, self.cv_img_original)
            if selector.result_rect is None:
                self.log_operation(f"❌ 取消操作: {method_name}")
                return 
            rect_roi = selector.result_rect
            self.log_operation(f"🖱️ 选区确定: {rect_roi}")

        # 情况C: 仅四点透视
        if config.get("interactive_points", False):
            selector = PointSelector(self.root, self.cv_img_original)
            if selector.result_points is None:
                self.log_operation(f"❌ 取消操作: {method_name}")
                return
            points = selector.result_points
            self.log_operation(f"🖱️ 四点确定: {points.tolist()}")
       
        # 参数弹窗

        # 检查是否需要参数弹窗
        config_params = config.get("params", [])
        
        if not config_params:
            # 如果配置中 params 为空列表 (例如 画布裁剪)，则直接使用空字典，不弹窗
            params = {}
        else:
            # 如果有参数，才弹出对话框
            history = self.param_history.get(method_name, {})
            dialog = MultiParamDialog(self.root, f"参数: {method_name}", config_params, history_values=history)
            if dialog.result_data is None: return 
            params = dialog.result_data
        
        # =========================================================
        # 第二阶段：线程准备与启动
        # (这里是修改的核心：不再直接运行，而是打包给线程)
        # =========================================================
        
        # 1. 更新 UI 状态：显示“正在处理”，让鼠标转圈，启动进度条
        ui_msg = f"🚀 正在后台执行: {method_name}..."
        self.log_operation(ui_msg)
        self.root.config(cursor="watch")      # 鼠标变成沙漏/忙碌状态
        self.progress_bar.pack(fill=tk.X, padx=2, pady=(0, 5), side=tk.BOTTOM) # 显示进度条
        self.progress_bar.start(10)           # 进度条开始跑动
        
        # 2. 准备数据 (主线程的数据要在此时复制一份，防止线程冲突)
        img_in = self.cv_img_original.copy()
        kwargs = params.copy()
        
        # 把刚才交互获取到的坐标塞进去
        if rect_roi: kwargs['rect'] = rect_roi
        if points is not None: kwargs['points'] = points
        if points_relative is not None: kwargs['points_relative'] = points_relative
        
        # 【新增】把暂存的颜色列表塞进去
        if target_colors is not None: kwargs['target_colors'] = target_colors

        # 3. 定义后台干活的工人 (Worker)
        def worker_thread():
            try:
                # --- 这里是最耗时的步骤 ---
                res = config["func"](img_in, **kwargs)
                
                # --- 算完后，告诉主线程 (成功) ---
                # 注意：不能在这里直接 self.cv_img_processed = res，必须用 root.after
                self.root.after(0, lambda: self.on_processing_finished(res, method_name, params, None))
                
            except Exception as e:
                # --- 出错后，告诉主线程 (失败) ---
                err_msg = str(e) 
                self.root.after(0, lambda: self.on_processing_finished(None, method_name, params, err_msg))
                # self.root.after(0, lambda: self.on_processing_finished(None, method_name, params, str(e)))

        # 4. 启动线程
        t = threading.Thread(target=worker_thread)
        t.daemon = True # 设置守护线程，主程序关闭时它也会自动停止
        t.start()

    # === 【新增】任务完成后的回调函数 ===
    def on_processing_finished(self, result_image, method_name, params, error_msg=None):
        """此函数由主线程调用，用于更新 UI"""
        self.progress_bar.stop()          # 停止动画
        self.progress_bar.pack_forget()   # 隐藏进度条 (或者不隐藏，看你喜好)
        self.root.config(cursor="")       # 恢复鼠标指针

        if error_msg:
            messagebox.showerror("算法错误", error_msg)
            self.log_operation(f"❌ 失败: {error_msg}")
            self.pending_log_entry = None
        else:
            self.cv_img_processed = result_image
            self.param_history[method_name] = params
            self.refresh_display()
            self.log_operation(f"🎉 处理完成: {method_name}")
            self.pending_log_entry = f"应用算法: [{method_name}] | 参数: {params}"

    def save_image(self):
        if self.cv_img_processed is None: return
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            cv2.imencode(".jpg", self.cv_img_processed)[1].tofile(path)
            
            self.cv_img_original = self.cv_img_processed.copy()
            
            if self.pending_log_entry:
                self.log_operation(f"✅ 确认应用并保存: {self.pending_log_entry}", is_ui_only=False)
                self.pending_log_entry = None 
            
            save_msg = f"💾 文件已保存至: {os.path.basename(path)}"
            self.log_operation(save_msg, is_ui_only=False)
            
            self.refresh_display()

    # (显示逻辑)
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
