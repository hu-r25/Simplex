import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات الصفحة والتنسيق الاحترافي ---
st.set_page_config(page_title="Simplex Solver Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-header { font-size: 2.2rem; color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 25px; }
    
    /* تنظيف خانات الإدخال */
    button.step-up, button.step-down { display: none !important; }
    input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
    input[type=number] { -moz-appearance: textfield; text-align: center; font-size: 18px !important; }

    /* صندوق المعادلات القياسية المحسن */
    .standard-form-box { 
        background-color: #161b22; 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #30363d; 
        margin: 20px 0;
    }
    .math-line { 
        font-family: 'Consolas', monospace;
        color: #ffcc00;
        font-size: 1.2rem;
        margin-bottom: 10px;
        display: block;
        direction: ltr;
        text-align: left;
    }
    .math-label { color: #58a6ff; font-weight: bold; margin-right: 15px; direction: rtl; display: inline-block; min-width: 100px; }
    .divider-line { border-top: 1px solid #30363d; margin: 15px 0; }

    /* تنسيق كروت المتغيرات */
    .variable-card {
        background-color: #161b22; padding: 8px 15px; border-radius: 8px;
        display: flex; align-items: center; gap: 12px; border: 1px solid #30363d;
        min-width: 180px; justify-content: center;
    }
    .var-tag { background-color: #21262d; color: #58a6ff; padding: 2px 10px; border-radius: 5px; font-weight: bold; }
    
    .pivot-bar-container { display: flex; justify-content: space-around; align-items: center; padding: 20px 0; }
    .stTable { width: 100%; border: 1px solid #30363d !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>🛡️ Simplex Solver Pro</div>", unsafe_allow_html=True)

# --- 2. الإعدادات والمدخلات ---
col_s1, col_s2 = st.columns(2)
with col_s1: n_vars = st.selectbox("عدد المتغيرات الأصلية (X):", [2, 3, 4], index=0)
with col_s2: n_const = st.selectbox("عدد القيود المتاحة:", [2, 3, 4], index=0)

st.divider()

c_obj, c_const = st.columns([1, 1.3], gap="large")
with c_obj:
    st.subheader("🎯 دالة الهدف (Z)")
    obj_coeffs = [float(st.number_input(f"X{i+1}", value=0, step=1, format="%d", key=f"obj_{i}")) for i in range(n_vars)]

with c_const:
    st.subheader("⛓️ القيود الهيكلية (≤)")
    constraints_matrix = []
    rhs_values = []
    for i in range(n_const):
        r_cols = st.columns(n_vars + 1)
        row = [float(r_cols[j].number_input(f"L{i+1}-X{j+1}", value=0, step=1, format="%d", key=f"c_{i}_{j}", label_visibility="collapsed")) for j in range(n_vars)]
        rhs_values.append(float(r_cols[-1].number_input(f"RHS {i+1}", value=0, step=1, format="%d", key=f"rhs_{i}", label_visibility="collapsed")))
        constraints_matrix.append(row)

if st.button("🚀 بدأ التحليل الرياضي الشامل", use_container_width=True):
    # --- المرحلة 1: إعادة معادلات القيود مع الفاصل الرفيع ---
    st.subheader("1️⃣ الصيغة القياسية النهائية (Standard Form)")
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    obj_final_text = " + ".join([f"{int(cj_full[idx])}{col_names[idx]}" for idx in range(len(col_names))])
    
    html_content = "<div class='standard-form-box'>"
    html_content += f"<div class='math-line'><span class='math-label'>دالة الهدف:</span> Max Z = {obj_final_text}</div>"
    html_content += "<div class='divider-line'></div>" # الخط الرفيع الفاصل
    html_content += "<span class='math-label' style='display:block; margin-bottom:12px;'>القيود المحولة:</span>"
    for i in range(n_const):
        eq_text = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)])
        eq_text += f" + 1{s_vars[i]} = {int(rhs_values[i])}"
        html_content += f"<div class='math-line' style='padding-left:30px;'>المعادلة {i+1} : {eq_text}</div>"
    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)

    # --- تهيئة مصفوفة السمبلكس ---
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 8):
        st.markdown(f"### 📍 جدول السمبلكس الموحد - التكرار رقم ({it})")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # 1. عرض Cj فوق الجدول
        st.table(pd.DataFrame([cj_full.astype(int)], columns=col_names, index=["Cj"]))

        # 2. بناء الجدول الموحد (المشترك) كما في الصورة المطلوبة
        combined_data = []
        p_col_idx = np.argmin(deltas)
        
        # إضافة صفوف الأساس
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            combined_data.append([basis[i], int(cb[i]), f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"])
        
        # إضافة صف Zj المشترك
        combined_data.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        
        # إضافة صف Zj - Cj المشترك
        combined_data.append(["Zj - Cj", "", ""] + [f"{val:.2f}" for val in deltas] + ["-"])

        full_cols = ["Basis", "CB", "XB"] + col_names + ["Ratio"]
        st.table(pd.DataFrame(combined_data, columns=full_cols))

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 تم الوصول للحل الأمثل: Z = {current_z:.2f}")
            break
            
        # 3. شريط الارتكاز المنظم
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        st.markdown(f"""
            <div class="pivot-bar-container">
                <div class="variable-card"><span style="font-size:1.3rem;">📥</span><span>الداخل:</span><span class="var-tag">{col_names[p_col_idx]}</span></div>
                <div style="display:flex; align-items:center; gap:10px;"><span style="font-size:1.3rem;">🎯</span><span>الارتكاز:</span><span style="background:#21262d; color:#3fb950; padding:4px 10px; border-radius:5px; font-weight:bold;">{matrix[p_row_idx, p_col_idx]:.2f}</span></div>
                <div class="variable-card"><span style="font-size:1.3rem;">📤</span><span>الخارج:</span><span class="var-tag">{basis[p_row_idx]}</span></div>
            </div>
        """, unsafe_allow_html=True)
        st.divider()

        # تحديث مصفوفة الحل
        pivot_val = matrix[p_row_idx, p_col_idx]
        matrix[p_row_idx] /= pivot_val
        xb[p_row_idx] /= pivot_val
        for i in range(n_const):
            if i != p_row_idx:
                factor = matrix[i, p_col_idx]
                matrix[i] -= factor * matrix[p_row_idx]
                xb[i] -= factor * xb[p_row_idx]
        basis[p_row_idx] = col_names[p_col_idx]
        cb[p_row_idx] = cj_full[p_col_idx]
