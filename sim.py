import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات الصفحة والتنسيق الاحترافي ---
st.set_page_config(page_title="Simplex Solver Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-header { font-size: 2.2rem; color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 20px; }
    
    /* تنظيف خانات الإدخال */
    button.step-up, button.step-down { display: none !important; }
    input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
    input[type=number] { -moz-appearance: textfield; text-align: center; font-size: 16px !important; }

    /* تنسيق صندوق المعادلات النهائية */
    .final-equation-box { 
        background-color: #161b22; 
        padding: 25px; 
        border-radius: 15px; 
        border: 2px solid #30363d; 
        margin: 20px 0;
        text-align: left;
    }
    .math-line { 
        font-family: 'Consolas', monospace;
        color: #ffcc00;
        font-size: 1.2rem;
        margin-bottom: 10px;
        display: block;
        direction: ltr;
    }
    .math-label { color: #58a6ff; font-weight: bold; margin-right: 15px; direction: rtl; display: inline-block; }
    
    .pivot-bar { background-color: #1f2937; padding: 10px; border-radius: 8px; border-left: 5px solid #ffcc00; color: #ffcc00; margin: 10px 0; font-weight: bold; }
    .stTable { width: 100%; border: 1px solid #30363d !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>🛡️ Simplex Solver Pro</div>", unsafe_allow_html=True)

# --- 2. إعدادات المسألة ---
col_s1, col_s2 = st.columns(2)
with col_s1:
    n_vars = st.selectbox("عدد المتغيرات الأصلية (X):", [2, 3, 4], index=0)
with col_s2:
    n_const = st.selectbox("عدد القيود:", [2, 3, 4], index=0)

st.divider()

# --- 3. مدخلات دالة الهدف والقيود ---
c_obj, c_const = st.columns([1, 1.3], gap="medium")

with c_obj:
    st.subheader("🎯 دالة الهدف (Z)")
    obj_coeffs = []
    cols = st.columns(n_vars)
    for i in range(n_vars):
        with cols[i]:
            val = st.number_input(f"X{i+1}", value=0, step=1, format="%d", key=f"obj_{i}")
            obj_coeffs.append(float(val))

with c_const:
    st.subheader("⛓️ القيود (≤)")
    constraints_matrix = []
    rhs_values = []
    for i in range(n_const):
        r_cols = st.columns(n_vars + 1)
        row = []
        for j in range(n_vars):
            with r_cols[j]:
                val = st.number_input(f"L{i+1}-X{j+1}", value=0, step=1, format="%d", key=f"c_{i}_{j}", label_visibility="collapsed")
                row.append(float(val))
        with r_cols[-1]:
            rhs = st.number_input(f"RHS {i+1}", value=0, step=1, format="%d", key=f"rhs_{i}", label_visibility="collapsed")
            rhs_values.append(float(rhs))
        constraints_matrix.append(row)

if st.button("🚀 بدأ التحليل الرياضي الشامل", use_container_width=True):
    # --- الصيغة القياسية الكاملة ---
    st.subheader("1️⃣ الصيغة القياسية النهائية (Standard Form)")
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    # بناء دالة الهدف النهائية
    obj_final_text = " + ".join([f"{int(cj_full[idx])}{col_names[idx]}" for idx in range(len(col_names))])
    
    # عرض كل شيء في صندوق واحد مرتب
    html_content = "<div class='final-equation-box'>"
    html_content += f"<div class='math-line'><span class='math-label'>دالة الهدف:</span> Max Z = {obj_final_text}</div><br>"
    html_content += "<span class='math-label' style='display:block; margin-bottom:10px;'>القيود المحولة:</span>"
    
    for i in range(n_const):
        eq_text = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)])
        eq_text += f" + 1{s_vars[i]} = {int(rhs_values[i])}"
        html_content += f"<div class='math-line' style='margin-left:20px;'>المعادلة {i+1} : {eq_text}</div>"
    
    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)

    # --- تهيئة المصفوفة والحل ---
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 8):
        st.markdown(f"### 📍 جدول السمبلكس - التكرار رقم ({it})")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # عرض Cj
        cj_display = pd.DataFrame([cj_full.astype(int)], columns=col_names, index=["Cj"])
        st.write("**توزيع معاملات Cj فوق المتغيرات:**")
        st.table(cj_display)

        # جدول البيانات الرئيسي
        main_table_data = []
        p_col_idx = np.argmin(deltas)
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            row = [basis[i], int(cb[i]), f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"]
            main_table_data.append(row)
        
        df_main = pd.DataFrame(main_table_data, columns=["Basis", "CB", "XB"] + col_names + ["Ratio"])
        st.table(df_main)

        # صفوف Zj و Zj-Cj الأفقية
        footer_data = [
            ["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + [""],
            ["Zj - Cj", "", ""] + [f"{val:.2f}" for val in deltas] + [""]
        ]
        df_footer = pd.DataFrame(footer_data, columns=["Basis", "CB", "XB"] + col_names + ["Ratio"])
        st.table(df_footer)

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 تم الوصول للحل الأمثل: Z = {current_z:.2f}")
            break
            
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        st.markdown(f"<div class='pivot-bar'>🎯 الارتكاز: {matrix[p_row_idx, p_col_idx]:.2f} | ⬅️ الداخل: {col_names[p_col_idx]} | ➡️ الخارج: {basis[p_row_idx]}</div>", unsafe_allow_html=True)
        st.divider()

        # تحديث المصفوفة
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
