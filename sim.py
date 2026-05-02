import customtkinter as ctk
import numpy as np
from tkinter import messagebox

# إعدادات المظهر
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FinalSimplexApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Simplex Solver Pro - Basra University Edition")
        self.geometry("1200x950")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. لوحة التحكم (Control Panel)
        self.control_panel = ctk.CTkFrame(self, height=150, fg_color="#1a1a1a")
        self.control_panel.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        
        ctk.CTkLabel(self.control_panel, text="إعداد مسألة البرمجة الخطية", font=("Arial", 20, "bold"), text_color="#3b8ed0").pack(pady=10)
        
        ctrl_inner = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        ctrl_inner.pack()

        ctk.CTkLabel(ctrl_inner, text="عدد المتغيرات:").grid(row=0, column=0, padx=10)
        self.var_select = ctk.CTkSegmentedButton(ctrl_inner, values=["2", "3", "4"], command=self.refresh_inputs)
        self.var_select.set("2")
        self.var_select.grid(row=0, column=1, padx=10)

        ctk.CTkLabel(ctrl_inner, text="عدد القيود:").grid(row=0, column=2, padx=10)
        self.const_select = ctk.CTkSegmentedButton(ctrl_inner, values=["2", "3", "4"], command=self.refresh_inputs)
        self.const_select.set("2")
        self.const_select.grid(row=0, column=3, padx=10)

        # 2. منطقة العمل (إدخال + جداول)
        self.work_area = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.work_area.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        # 3. شريط التنفيذ السفلي
        self.bottom_frame = ctk.CTkFrame(self, height=80, fg_color="#1a1a1a")
        self.bottom_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=20)
        
        self.solve_btn = ctk.CTkButton(self.bottom_frame, text="🚀 تحليل واستخراج النتائج", 
                                       command=self.solve_and_render, fg_color="#2eb85c", 
                                       hover_color="#1e7e34", font=("Arial", 16, "bold"), height=45)
        self.solve_btn.pack(pady=15)

        self.refresh_inputs()

    def refresh_inputs(self, _=None):
        for widget in self.work_area.winfo_children():
            widget.destroy()
        
        num_v = int(self.var_select.get())
        num_c = int(self.const_select.get())

        # دالة الهدف
        obj_frame = ctk.CTkFrame(self.work_area)
        obj_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(obj_frame, text="🎯 دالة الهدف (Maximization Z):", font=("Arial", 15, "bold")).pack(pady=10)
        
        self.obj_entries = []
        row_obj = ctk.CTkFrame(obj_frame, fg_color="transparent")
        row_obj.pack(pady=5)
        for i in range(num_v):
            ctk.CTkLabel(row_obj, text=f"X{i+1}").pack(side="left", padx=2)
            ent = ctk.CTkEntry(row_obj, width=75, justify="center", border_color="#3b8ed0")
            ent.pack(side="left", padx=10)
            self.obj_entries.append(ent)

        # القيود
        ctk.CTkLabel(self.work_area, text="⛓️ القيود (Constraints):", font=("Arial", 15, "bold")).pack(pady=15)
        self.const_rows = []
        for i in range(num_c):
            f = ctk.CTkFrame(self.work_area)
            f.pack(pady=5, fill="x")
            current_row_ents = []
            for j in range(num_v):
                ent = ctk.CTkEntry(f, width=75, justify="center", placeholder_text=f"X{j+1}")
                ent.pack(side="left", padx=10, pady=10)
                current_row_ents.append(ent)
            
            ctk.CTkLabel(f, text="≤", font=("Arial", 20, "bold")).pack(side="left", padx=15)
            rhs = ctk.CTkEntry(f, width=90, fg_color="#2b2b2b", border_color="#e55353", justify="center")
            rhs.pack(side="left", padx=10)
            self.const_rows.append((current_row_ents, rhs))

        self.tables_container = ctk.CTkFrame(self.work_area, fg_color="transparent")
        self.tables_container.pack(fill="x", pady=20)

    def draw_tableau(self, data, title, pivot_pos=None):
        card = ctk.CTkFrame(self.tables_container, border_width=1, border_color="#444")
        card.pack(fill="x", pady=15)
        ctk.CTkLabel(card, text=title, font=("Arial", 16, "bold"), text_color="#ffcc00").pack(pady=10)
        
        t_frame = ctk.CTkFrame(card, fg_color="#111")
        t_frame.pack(padx=15, pady=15)

        # Cj Row
        ctk.CTkLabel(t_frame, text="Cj", text_color="#888").grid(row=0, column=2)
        for j, val in enumerate(data['cj']):
            ctk.CTkLabel(t_frame, text=f"{val}", fg_color="#222", width=75).grid(row=0, column=3+j, padx=1)

        # Headers
        cols = ["Basic", "CB", "XB"] + data['col_names'] + ["Ratio"]
        for j, col in enumerate(cols):
            ctk.CTkLabel(t_frame, text=col, font=("Arial", 12, "bold"), text_color="#3b8ed0").grid(row=1, column=j, padx=8, pady=8)

        # Matrix Body
        for i in range(len(data['basis'])):
            ctk.CTkLabel(t_frame, text=data['basis'][i], font=("Arial", 11, "bold")).grid(row=2+i, column=0)
            ctk.CTkLabel(t_frame, text=f"{data['cb'][i]}").grid(row=2+i, column=1)
            ctk.CTkLabel(t_frame, text=f"{data['xb'][i]:.2f}", text_color="#00ff00").grid(row=2+i, column=2, padx=5)
            
            for j in range(len(data['col_names'])):
                val = data['matrix'][i][j]
                is_pivot = pivot_pos and pivot_pos == (i, j)
                txt = f"[{val:.2f}]" if is_pivot else f"{val:.2f}"
                lbl = ctk.CTkLabel(t_frame, text=txt, text_color="yellow" if is_pivot else "white")
                lbl.grid(row=2+i, column=3+j)
            
            r_val = data['ratios'][i]
            r_txt = f"{r_val:.2f}" if r_val != float('inf') else "-"
            ctk.CTkLabel(t_frame, text=r_txt).grid(row=2+i, column=3+len(data['col_names']))

        # Bottom Δj Row
        last_row = 2 + len(data['basis'])
        ctk.CTkLabel(t_frame, text=f"Z = {data['z']:.2f}", font=("Arial", 13, "bold"), text_color="#ffcc00").grid(row=last_row, column=2, pady=10)
        for j, d in enumerate(data['deltas']):
            txt = f"Δ{j+1}={d:.2f}" + (" ↑" if d < 0 else "")
            ctk.CTkLabel(t_frame, text=txt, text_color="#2eb85c" if d < 0 else "white").grid(row=last_row, column=3+j)

    def show_final_summary(self, z, variables, col_names):
        summary_card = ctk.CTkFrame(self.tables_container, fg_color="#2b2b2b", border_width=2, border_color="#2eb85c")
        summary_card.pack(fill="x", pady=30, padx=50)
        
        ctk.CTkLabel(summary_card, text="📊 النتيجة النهائية للحل الأمثل", font=("Arial", 22, "bold"), text_color="#2eb85c").pack(pady=15)
        
        res_text = f"القيمة العظمى لدالة الهدف: Z = {z:.2f}\n"
        res_text += "--------------------------------------\n"
        
        # استخراج قيم المتغيرات الأصلية فقط (X1, X2...)
        for i, name in enumerate(col_names):
            if name.startswith('X'):
                val = variables[i]
                res_text += f"قيمة المتغير {name} = {val:.2f}\n"

        ctk.CTkLabel(summary_card, text=res_text, font=("Consolas", 18), justify="left").pack(pady=20)

    def solve_and_render(self):
        try:
            for w in self.tables_container.winfo_children(): w.destroy()

            c = np.array([float(e.get()) for e in self.obj_entries])
            A, b = [], []
            for row_ents, rhs_ent in self.const_rows:
                A.append([float(e.get()) for e in row_ents])
                b.append(float(rhs_ent.get()))
            
            A, b = np.array(A), np.array(b)
            n_vars, n_const = len(c), len(A)

            cj_full = np.concatenate([c, [0]*n_const])
            matrix = np.hstack([A, np.eye(n_const)])
            xb = b.astype(float)
            basis = [f"S{i+1}" for i in range(n_const)]
            cb = np.zeros(n_const)
            col_names = [f"X{i+1}" for i in range(n_vars)] + [f"S{i+1}" for i in range(n_const)]

            final_z = 0
            final_vars = np.zeros(len(col_names))

            for it in range(10): # Iterations
                z = np.dot(cb, xb)
                deltas = [np.dot(cb, matrix[:, j]) - cj_full[j] for j in range(len(cj_full))]
                
                # تخزين النتائج الحالية كأفضل نتائج حتى الآن
                final_z = z
                current_vars = np.zeros(len(col_names))
                for i, b_name in enumerate(basis):
                    idx = col_names.index(b_name)
                    current_vars[idx] = xb[i]
                final_vars = current_vars

                p_col = np.argmin(deltas)
                ratios = [xb[i]/matrix[i, p_col] if matrix[i, p_col] > 0 else float('inf') for i in range(n_const)]
                p_row = np.argmin(ratios)

                self.draw_tableau({
                    'cj': cj_full, 'col_names': col_names, 'basis': basis.copy(),
                    'cb': cb.copy(), 'xb': xb.copy(), 'matrix': matrix.copy(),
                    'z': z, 'deltas': deltas, 'ratios': ratios
                }, f"الجدول التكراري رقم {it} - (Simplex Tableau)", pivot_pos=(p_row, p_col) if min(deltas) < 0 else None)

                if min(deltas) >= 0: break

                # تحديث المصفوفة
                p_val = matrix[p_row, p_col]
                matrix[p_row] /= p_val
                xb[p_row] /= p_val
                for i in range(n_const):
                    if i != p_row:
                        factor = matrix[i, p_col]
                        matrix[i] -= factor * matrix[p_row]
                        xb[i] -= factor * xb[p_row]
                
                basis[p_row] = col_names[p_col]
                cb[p_row] = cj_full[p_col]

            # عرض الملخص النهائي
            self.show_final_summary(final_z, final_vars, col_names)

        except Exception as e:
            messagebox.showerror("خطأ في البيانات", "يرجى التأكد من ملء جميع الخلايا بأرقام صحيحة قبل البدء.")

if __name__ == "__main__":
    app = FinalSimplexApp()
    app.mainloop()