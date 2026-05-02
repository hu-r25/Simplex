import streamlit as st
import numpy as np
import pandas as pd

# إعدادات الصفحة
st.set_page_config(page_title="Simplex Solver Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .steps-container { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; font-family: monospace; color: #c9d1d9; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Simplex Solver Pro - Basra University Edition")

# إدخال عدد المتغيرات والقيود
col_setup1, col_setup2 = st.columns(2)
with col_setup1:
    n_vars = st.selectbox("عدد المتغيرات الأصلية (X):", [2, 3, 4], index=0)
with col_setup2:
    n_const = st.selectbox("عدد القيود:", [2, 3, 4], index=0)

# دالة الهدف
st.subheader("🎯 دالة الهدف (Maximization Z)")
obj_coeffs = []
cols_obj = st.columns(n_vars)
for i in range(n_vars):
    with cols_obj[i]:
        val = st.number_input(f"معامل X{i+1}", value=0.0, step=1.0, format="%f", key=f"obj_{i}")
        obj_coeffs.append(val)

# القيود
st.subheader("⛓️ القيود (Constraints ≤)")
constraints_matrix = []
rhs_values = []
for i in range(n_const):
    cols_c = st.columns(n_vars + 1)
    row = []
    for j in range(n_vars):
        with cols_c[j]:
            val = st.number_input(f"L{i+1}-X{j+1}", value=0.0, step=1.0, key=f"c_{i}_{j}")
            row.append(val)
    with cols_c[-1]:
        rhs = st.number_input(f"ناتج القيد {i+1}", value=0.0, step=1.0, key=f"rhs_{i}")
        rhs_values.append(rhs)
    constraints_matrix.append(row)

if st.button("🚀 بدأ التحليل الرياضي الشامل"):
    # خطوة التحويل للصيغة القياسية
    st.subheader("1️⃣ تحويل المسألة إلى الصيغة القياسية (Standard Form)")
    s_vars = [f"S{i+1}" for i in range(n_const)]
    obj_str = " + ".join([f"{obj_coeffs[i]}X{i+1}" for i in range(n_vars)]) + " + " + " + ".join([f"0{s}" for s in s_vars])
    st.info(f"Max Z = {obj_str}")
    
    for i in range(n_const):
        eq = " + ".join([f"{constraints_matrix[i][j]}X{j+1}" for j in range(n_vars)]) + f" + {s_vars[i]} = {rhs_values[i]}"
        st.write(f"المعادلة {i+1}: {eq}")

    # تحضير المصفوفة الابتدائية
    col_names = [f"X{i+1}" for i in range(n_vars)] + [f"S{i+1}" for i in range(n_const)]
    cj_full = np.concatenate([obj_coeffs, [0]*n_const])
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    # دورات السمبلكس
    for it in range(1, 11):
        z_val = np.dot(cb, xb)
        deltas = [np.dot(cb, matrix[:, j]) - cj_full[j] for j in range(len(cj_full))]
        
        st.markdown(f"### 📍 جدول السمبلكس - التكرار رقم ({it})")
        
        # عرض الجدول
        df_display = pd.DataFrame(matrix, columns=col_names, index=basis)
        df_display.insert(0, "CB", cb)
        df_display.insert(1, "XB", xb)
        
        # حساب النسب لتحديد الارتكاز
        p_col = np.argmin(deltas)
        ratios = []
        for i in range(n_const):
            if matrix[i, p_col] > 0: ratios.append(xb[i] / matrix[i, p_col])
            else: ratios.append(np.inf)
        
        df_display["Ratio"] = [f"{r:.2f}" if r != np.inf else "-" for r in ratios]
        st.table(df_display.style.format(precision=2))
        st.write(f"**قيمة Z الحالية:** {z_val:.2f}")

        if min(deltas) >= 0:
            st.success(f"🏁 تم الوصول للحل الأمثل! القيمة العظمى Z = {z_val:.2f}")
            res_summary = "القيم النهائية: " + ", ".join([f"{basis[i]} = {xb[i]:.2f}" for i in range(n_const)])
            st.write(res_summary)
            break
            
        # تحديث الصفوف (Gauss-Jordan)
        p_row = np.argmin(ratios)
        pivot_val = matrix[p_row, p_col]
        matrix[p_row] /= pivot_val
        xb[p_row] /= pivot_val
        for i in range(n_const):
            if i != p_row:
                factor = matrix[i, p_col]
                matrix[i] -= factor * matrix[p_row]
                xb[i] -= factor * xb[p_row]
        
        basis[p_row] = col_names[p_col]
        cb[p_row] = cj_full[p_col]
