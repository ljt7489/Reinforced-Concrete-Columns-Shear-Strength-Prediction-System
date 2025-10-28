import tkinter as tk
from tkinter import ttk, messagebox, font
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBClassifier
import joblib
import re
from PIL import Image, ImageTk
import sys
import os

def resource_path(relative_path):
    """ 获取资源的绝对路径，兼容开发环境和 PyInstaller 打包后的环境 """
    if hasattr(sys, '_MEIPASS'):
        # 如果存在 _MEIPASS 属性，说明程序是打包后的，资源在临时解压目录
        base_path = sys._MEIPASS
    else:
        # 否则是开发环境，资源在当前目录或相对路径下
        base_path = os.path.abspath(".")
    # 拼接并返回完整路径
    return os.path.join(base_path, relative_path)
class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shear Strength Prediction System for Reinforced Concrete Columns")
        self.root.geometry("1400x800")  # 进一步增加窗口大小以容纳图片

        # 创建Times New Roman字体
        self.custom_font = font.Font(family="Times New Roman", size=13)
        self.title_font = font.Font(family="Times New Roman", size=12, weight="bold")

        # 配置根窗口默认字体
        self.root.option_add("*Font", self.custom_font)

        # 加载模型
        try:
            self.catboost_model = CatBoostRegressor()
            self.catboost_model.load_model('catboost_model.cbm')
            self.xgb_model = joblib.load('xgb_model.pkl')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model: {str(e)}")
            return

        # 根据错误信息，模型期望的特征名称（包括n, λ, L/h）
        self.expected_feature_names = [
            "L", "b", "h", "d", "fc", "Ag", "pl%", "fy", "ps%", "Asl", "fyt", "s", "Ast", "P", "n", "λ", "L/h"
        ]

        # 输入特征名称与模型期望名称的映射
        self.feature_mapping = {
            "L(mm)": "L",
            "b(mm)": "b",
            "h(mm)": "h",
            "d(mm)": "d",
            "fc(mm)": "fc",
            "Ag(mm)": "Ag",
            "pl(%)": "pl%",
            "fy(Mpa)": "fy",
            "ps(%)": "ps%",
            "Asl(mm²)": "Asl",
            "fyt(Mpa)": "fyt",
            "s(mm)": "s",
            "Ast(mm²)": "Ast",
            "P(kN)": "P",
            "n": "n",
            "λ": "λ",
            "L/h": "L/h"
        }

        # 显示在界面上的特征名称（使用完整的英文描述）
        self.display_feature_names = [
            "L(mm)", "b(mm)", "h(mm)", "d(mm)", "fc(mm)", "Ag(mm)", "pl(%)",
            "fy(Mpa)", "ps(%)", "Asl(mm²)", "fyt(Mpa)", "s(mm)", "Ast(mm²)", "P(kN)",
            "n", "λ", "L/h"
        ]

        # 特征名称到英文描述的映射
        self.feature_descriptions = {
            "L(mm)": "Height of the column (mm)",
            "b(mm)": "Width of column section (mm)",
            "h(mm)": "Height of column section (mm)",
            "d(mm)": "Section effective depth (mm)",
            "fc(mm)": "Compressive strength of concrete (mm)",
            "Ag(mm)": "Gross area of column section (mm)",
            "pl(%)": "Longitudinal reinforcement ratio (%)",
            "fy(Mpa)": "Yield strength of longitudinal bar (Mpa)",
            "ps(%)": "Transverse reinforcement volumetric ratio (%)",
            "Asl(mm²)": "Area of a single longitudinal bar (mm²)",
            "fyt(Mpa)": "Yield strength of transverse steel (Mpa)",
            "s(mm)": "Spacing of transverse reinforcement (mm)",
            "Ast(mm²)": "Area of transverse reinforcement bar (mm²)",
            "P(kN)": "Applied axial load (kN)",
            "n": "Axial compression ratio",
            "λ": "Shear span ratio",
            "L/h": "Span-to-Depth Ratio"
        }

        # 创建主框架，分为左右两部分
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 左侧框架（输入参数）
        self.left_frame = ttk.Frame(self.main_frame, width=700)
        self.left_frame.pack(side="left", fill="both", expand=True)

        # 右侧框架（图片显示）
        self.right_frame = ttk.LabelFrame(self.main_frame, text="Schematic Diagram", width=500)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self.create_input_widgets()
        self.create_result_widgets()
        self.create_image_widget()  # 创建图片显示部件
        self.create_logo_widget()  # 创建logo显示部件

    def create_logo_widget(self):
        """创建并放置logo图片在右下角"""
        try:
            # 加载logo图片
            logo_path = resource_path("picture/file/Njfu_logo.png")  # 请确保路径正确
            logo_image = Image.open(logo_path)

            # 调整logo大小，例如设置最大宽度或高度
            logo_image.thumbnail((100, 100), Image.LANCZOS)  # 按需调整尺寸

            # 转换为Tkinter可显示的格式
            self.logo_photo = ImageTk.PhotoImage(logo_image)

            # 创建标签显示logo，使用place布局管理器精确定位于右下角
            logo_label = ttk.Label(self.root, image=self.logo_photo)
            # 使用place管理器，设置相对x和y坐标（1.0代表右下角），锚点定位在右下角（SE）
            logo_label.place(relx=1.0, rely=1.0, anchor="se", x=-15, y=-15)  # 添加少量偏移避免紧贴边缘

        except Exception as e:
            # 如果图片加载失败，打印错误信息（或可根据需要添加更显眼的提示）
            print(f"Failed to load logo image: {str(e)}")
            # 可选：在界面上显示一个简单的错误文字标签
            error_label = ttk.Label(self.root, text="Logo not found", foreground="gray")
            error_label.place(relx=1.0, rely=1.0, anchor="se")

    def create_image_widget(self):
        """创建图片显示区域"""
        try:
            # 加载并显示图片
            image_path = resource_path("picture/file/A_schematic diagram_of_a_typical_RC_column.png")
            image = Image.open(image_path)

            # 调整图片大小以适应界面
            max_width, max_height = 650, 550
            image.thumbnail((max_width, max_height), Image.LANCZOS)

            # 转换为Tkinter可显示的格式
            self.photo = ImageTk.PhotoImage(image)

            # 创建标签显示图片
            image_label = ttk.Label(self.right_frame, image=self.photo)
            image_label.pack(pady=10)

            # 添加图片说明
            caption = ttk.Label(self.right_frame,
                                text="Schematic Diagram of a Typical RC Column",
                                font=("Times New Roman", 10, "italic"),
                                justify="center")
            caption.pack(pady=(0, 10))

        except Exception as e:
            # 如果图片加载失败，显示错误信息
            error_label = ttk.Label(self.right_frame,
                                    text=f"Failed to load image: {str(e)}",
                                    foreground="red")
            error_label.pack(pady=10)

    def create_input_widgets(self):
        input_frame = ttk.LabelFrame(self.left_frame, text="Input Parameters", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)

        # 设置LabelFrame的字体
        style = ttk.Style()
        style.configure("TLabelframe.Label", font=("Times New Roman", 10, "bold"))

        self.entries = {}
        for i, feature in enumerate(self.display_feature_names):
            row = i % 6  # 减少每列的特征数量以适应更长的标签
            column = i // 6 * 2

            # 使用完整的英文描述创建标签
            description = self.feature_descriptions.get(feature, feature)
            label = ttk.Label(input_frame, text=f"{description}:", font=("Times New Roman", 9))
            label.grid(row=row, column=column, padx=5, pady=2, sticky="e")

            entry = ttk.Entry(input_frame, width=15, font=("Times New Roman", 9))
            entry.grid(row=row, column=column + 1, padx=5, pady=2, sticky="w")
            self.entries[feature] = entry

        predict_btn = ttk.Button(input_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=6, column=0, columnspan=4, pady=10)

    def create_result_widgets(self):
        result_frame = ttk.LabelFrame(self.left_frame, text="Prediction Result", padding="10")
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        style = ttk.Style()
        style.configure("TLabelframe.Label", font=("Times New Roman", 10, "bold"))

        self.result_text = tk.Text(result_frame, height=10, width=70,
                                   font=("Times New Roman", 10),
                                   relief="solid", borderwidth=1)
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)

        # 配置文本样式标签
        self.result_text.tag_configure("bold_red", font=("Times New Roman", 10, "bold"), foreground="red")

        clear_btn = ttk.Button(result_frame, text="Clear Result", command=self.clear_results)
        clear_btn.pack(side="right", padx=5, pady=5)

    def validate_input(self, value):
        """验证输入是否为有效数字"""
        pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
        return re.match(pattern, value.strip()) is not None

    def predict(self):
        try:
            input_data = {}
            invalid_fields = []

            # 验证所有输入字段
            for feature, entry in self.entries.items():
                value = entry.get().strip()

                if not value:
                    messagebox.showwarning("Incomplete Input",
                                           f"Please fill in field: {self.feature_descriptions.get(feature, feature)}")
                    return

                if not self.validate_input(value):
                    invalid_fields.append(self.feature_descriptions.get(feature, feature))
                    continue

                input_data[feature] = float(value)

            if invalid_fields:
                messagebox.showerror(
                    "Invalid Input",
                    f"Please enter valid numbers in the following fields:\n" +
                    "\n".join(invalid_fields)
                )
                return

            # 将输入数据转换为模型期望的格式
            model_input = {}
            for display_name, value in input_data.items():
                if display_name in self.feature_mapping:
                    model_name = self.feature_mapping[display_name]
                    model_input[model_name] = value

            # 确保所有期望的特征都存在
            for expected_feature in self.expected_feature_names:
                if expected_feature not in model_input:
                    messagebox.showerror(
                        "Feature Error",
                        f"Missing expected feature: {expected_feature}"
                    )
                    return

            # 转换为DataFrame，按照模型期望的顺序排列特征
            input_df = pd.DataFrame([model_input], columns=self.expected_feature_names)

            # 使用XGBoost预测m值
            m_pred = self.xgb_model.predict(input_df)[0]

            if m_pred == 0:
                FM = 'Flexure failure'
            elif m_pred == 1:
                FM = 'Shear failure'
            else:
                FM = 'Flexure-shear failure'

            # 将m极速模式预测值添加到特征中
            input_df_with_m = input_df.copy()
            input_df_with_m['predicted_m'] = m_pred

            # 使用CatBoost预测V值
            v_pred = self.catboost_model.predict(input_df_with_m)[0]

            # 显示结果（使用英文描述）
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Prediction Results:\n")

            # 插入加粗红色的Failure mode行
            self.result_text.insert(tk.END, "Failure mode: ")
            self.result_text.insert(tk.END, f"{FM}\n", "bold_red")

            # 插入加粗红色的Ultimate shear strength行
            self.result_text.insert(tk.END, "Ultimate shear strength: ")
            self.result_text.insert(tk.END, f"{v_pred:.4f} kN\n\n", "bold_red")

            self.result_text.insert(tk.END, "Input Parameters:\n")
            for feature, value in input_data.items():
                description = self.feature_descriptions.get(feature, feature)
                self.result_text.insert(tk.END, f"{description}: {value}\n")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {str(e)}")

    def clear_results(self):
        self.result_text.delete(1.0, tk.END)
        for entry in self.entries.values():
            entry.delete(0, tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()