import tkinter as tk
from tkinter import filedialog, font as tkfont
from PIL import Image, ImageTk
import pandas as pd
from ultralytics import YOLO
from textwrap import wrap
import os

# Налаштування шляхів
MODEL_PATH = r'best.pt'
PRICES_CSV = r'rpc_prices.csv'


# Завантаження моделі та csv-файлу 
model = YOLO(MODEL_PATH)

price_df = pd.read_csv(PRICES_CSV, encoding="utf-8-sig")
price_df.columns = [c.strip() for c in price_df.columns]
price_df["sku_name"] = price_df["sku_name"].astype(str).str.replace('"', '', regex=False).str.strip()
price_df["ua_name"]  = price_df["ua_name"].astype(str).str.replace('"', '', regex=False).str.strip()
price_df["price"]    = pd.to_numeric(price_df["price"], errors="coerce").fillna(0.0)

price_dict = dict(zip(price_df["sku_name"], price_df["price"]))
ua_dict    = dict(zip(price_df["sku_name"], price_df["ua_name"]))

names    = model.names
n_classes = len(names) if isinstance(names, list) else len(names.keys())
price_map, ua_map = {}, {}

for i in range(n_classes):
    cls_name = (names[i] if isinstance(names, list) else names.get(i) or names.get(str(i)))
    cls_name = str(cls_name).strip()
    price_map[i] = float(price_dict.get(cls_name, 0.0))
    ua_map[i]    = ua_dict.get(cls_name, cls_name)

# Логіка розпізнавання 
def run_detection(image_path):
    results = model.predict(
        source=image_path, conf=0.1,
        save=True, save_txt=False, save_crop=False, show=False
    )
    result_dir     = results[0].save_dir
    result_img_path = os.path.join(result_dir, os.path.basename(image_path))
    return results, result_img_path

# Генерація чеку
def generate_check(results):
    counts = {}
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            counts[cls] = counts.get(cls, 0) + 1
    items  = []
    total  = 0.0
    for cls, qty in counts.items():
        price    = price_map.get(cls, 0.0)
        subtotal = qty * price
        total   += subtotal
        items.append({
            "name":     ua_map.get(cls, f"клас_{cls}"),
            "qty":      qty,
            "price":    price,
            "subtotal": subtotal,
        })
    return items, total

# Палітра кольорів 
BG_DARK      = "#0F1117"
BG_CARD      = "#1A1D27"
BG_PANEL     = "#13151F"
ACCENT       = "#4F8EF7"
ACCENT_LIGHT = "#7AADFF"
TEXT_MAIN    = "#E8ECF4"
TEXT_MUTED   = "#7A8099"
TEXT_DIM     = "#4A5068"
SUCCESS      = "#3DD68C"
DIVIDER      = "#252836"
HOVER_BTN    = "#3A6DD8"
BTN_BG       = "#4F8EF7"

# Головний клас програми 
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Система розпізнавання та підрахунку вартості товарів")
        self.root.state("zoomed")
        self.root.resizable(True, True)
        self.root.minsize(900, 580)
        self.root.configure(bg=BG_DARK)

        self._build_ui()

    # Побудова інтерфейсу 
    def _build_ui(self):
        # Верхня панель
        header = tk.Frame(self.root, bg=BG_PANEL, height=56)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        tk.Label(
            header, text="🛒  Система розпізнавання та підрахунку вартості товарів",
            bg=BG_PANEL, fg=TEXT_MAIN,
            font=("Segoe UI", 14, "bold"),
            padx=24
        ).pack(side=tk.LEFT, pady=12)

        tk.Frame(self.root, bg=ACCENT, height=2).pack(fill=tk.X)

        # Основна область 
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=18)

        # Ліва панель — зображення 
        left = tk.Frame(body, bg=BG_CARD, bd=0, relief=tk.FLAT)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # рядок 0 — заголовок,
        # рядок 1 — зона зображення, рядок 2 — кнопка
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Заголовок секції 
        hdr_frame = tk.Frame(left, bg=BG_CARD)
        hdr_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 10))
        tk.Label(hdr_frame, text="Товари", bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Frame(hdr_frame, bg=DIVIDER, height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0), pady=6)

        # Зона прев'ю 
        self._has_image = False
        self.img_frame = tk.Frame(left, bg="#12141E", cursor="hand2")
        self.img_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

        self.placeholder = tk.Label(
            self.img_frame,
            text="📁\n\nНатисніть, щоб обрати зображення",
            bg="#12141E", fg=TEXT_DIM,
            font=("Segoe UI", 12),
            justify=tk.CENTER,
            cursor="hand2"
        )
        self.placeholder.place(relx=.5, rely=.5, anchor=tk.CENTER)

        self.img_label = tk.Label(self.img_frame, bg="#12141E", cursor="hand2")

        # Скидання фото і чека
        self.close_btn = tk.Label(
            self.img_frame,
            text="✕",
            bg="#2A2D3E", fg=TEXT_MUTED,
            font=("Segoe UI", 11),
            cursor="hand2",
            width=2, padx=6, pady=4
        )
        self.close_btn.place_forget()
        self.close_btn.bind("<Button-1>", lambda e: self.reset_image())
        self.close_btn.bind("<Enter>",  lambda e: self.close_btn.config(bg="#3A3D52", fg=TEXT_MAIN))
        self.close_btn.bind("<Leave>",  lambda e: self.close_btn.config(bg="#2A2D3E", fg=TEXT_MUTED))

        def _on_enter(e):
            if not self._has_image:
                self.img_frame.config(bg="#1A1C2E")
                self.placeholder.config(bg="#1A1C2E", fg=ACCENT_LIGHT)

        def _on_leave(e):
            if not self._has_image:
                self.img_frame.config(bg="#12141E")
                self.placeholder.config(bg="#12141E", fg=TEXT_DIM)

        for widget in (self.img_frame, self.placeholder, self.img_label):
            widget.bind("<Button-1>", lambda e: self.upload_image())
            widget.bind("<Enter>",    _on_enter)
            widget.bind("<Leave>",    _on_leave)

        # Права панель — чек 
        right = tk.Frame(body, bg=BG_CARD, width=360)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right.pack_propagate(False)

        self._section_label(right, "Чек")

        divider = tk.Frame(right, bg=DIVIDER, height=1)
        divider.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(4, 0))

        self.total_frame = tk.Frame(right, bg=BG_CARD)
        self.total_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(4, 16))

        tk.Label(self.total_frame, text="Разом:", bg=BG_CARD,
                 fg=TEXT_MUTED, font=("Segoe UI", 11)).pack(side=tk.LEFT)
        self.total_var = tk.StringVar(value="—")
        tk.Label(self.total_frame, textvariable=self.total_var,
                 bg=BG_CARD, fg=SUCCESS,
                 font=("Segoe UI", 15, "bold")).pack(side=tk.RIGHT)

        scroll_area = tk.Frame(right, bg=BG_CARD)
        scroll_area.pack(fill=tk.BOTH, expand=True, padx=(16, 0), pady=(0, 4))

        scrollbar = tk.Scrollbar(scroll_area, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.check_canvas = tk.Canvas(
            scroll_area, bg=BG_CARD, bd=0,
            highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.check_canvas.yview)
        self.check_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.items_frame = tk.Frame(self.check_canvas, bg=BG_CARD)
        self.canvas_window = self.check_canvas.create_window(
            (0, 0), window=self.items_frame, anchor="nw"
        )
        self.items_frame.bind("<Configure>", self._on_frame_configure)
        self.check_canvas.bind("<Configure>", self._on_canvas_configure)

    # Допоміжні методи UI 
    def _section_label(self, parent, text):
        f = tk.Frame(parent, bg=BG_CARD)
        f.pack(fill=tk.X, padx=16, pady=(14, 10))
        tk.Label(f, text=text, bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Frame(f, bg=DIVIDER, height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0), pady=6)

    def _on_frame_configure(self, event):
        self.check_canvas.configure(
            scrollregion=self.check_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.check_canvas.itemconfig(self.canvas_window, width=event.width)

    def _clear_items(self):
        for w in self.items_frame.winfo_children():
            w.destroy()

    # Завантаження та обробка зображення 
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Оберіть зображення",
            filetypes=[("Зображення", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        self.placeholder.config(text="⏳  Розпізнавання...")
        self.root.update()

        results, result_img_path = run_detection(file_path)
        self.display_image(result_img_path)

        items, total = generate_check(results)
        self.display_check(items, total)

    def display_image(self, img_path):
        self._has_image = True
        self.placeholder.place_forget()
        self.close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=6)
        self.close_btn.lift()
        self.root.update_idletasks()
        fw = max(self.img_frame.winfo_width()  - 4, 200)
        fh = max(self.img_frame.winfo_height() - 4, 200)
        img = Image.open(img_path)
        img.thumbnail((fw, fh))
        self.img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.img_tk)
        self.img_label.image = self.img_tk
        self.img_label.place(relx=.5, rely=.5, anchor=tk.CENTER)

    def reset_image(self):
        """Скидання фото і чекф до початкового стану"""
        self._has_image = False
        self.img_label.config(image="")
        self.img_label.image = None
        self.img_label.place_forget()
        self.close_btn.place_forget()
        self.img_frame.config(bg="#12141E")
        self.placeholder.config(
            text="📁\n\nНатисніть, щоб обрати зображення",
            bg="#12141E", fg=TEXT_DIM
        )
        self.placeholder.place(relx=.5, rely=.5, anchor=tk.CENTER)
        self._clear_items()
        self.total_var.set("—")

    def display_check(self, items, total):
        self._clear_items()

        if not items:
            tk.Label(self.items_frame, text="Товарів не знайдено",
                     bg=BG_CARD, fg=TEXT_DIM,
                     font=("Segoe UI", 10)).pack(pady=20)
            self.total_var.set("0.00 ₴")
            return

        for idx, item in enumerate(items):
            row_bg = BG_CARD if idx % 2 == 0 else "#1E2133"
            row = tk.Frame(self.items_frame, bg=row_bg)
            row.pack(fill=tk.X, pady=1)

            # Назва товару
            name_label = tk.Label(
                row, text=item["name"],
                bg=row_bg, fg=TEXT_MAIN,
                font=("Segoe UI", 10),
                wraplength=200, justify=tk.LEFT, anchor="w"
            )
            name_label.grid(row=0, column=0, sticky="w",
                            padx=(10, 4), pady=6)

            # К-сть × ціна
            mid_text = f"{item['qty']} × {item['price']:.2f} ₴"
            tk.Label(
                row, text=mid_text,
                bg=row_bg, fg=TEXT_MUTED,
                font=("Segoe UI", 9)
            ).grid(row=1, column=0, sticky="w", padx=(10, 4), pady=(0, 6))

            # Підсумок по позиції
            tk.Label(
                row, text=f"{item['subtotal']:.2f} ₴",
                bg=row_bg, fg=ACCENT_LIGHT,
                font=("Segoe UI", 11, "bold")
            ).grid(row=0, column=1, rowspan=2,
                   sticky="e", padx=(4, 12))

            row.columnconfigure(0, weight=1)

        self.total_var.set(f"{total:.2f} ₴")



if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.mainloop()