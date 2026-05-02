import streamlit as st
import numpy as np
import pandas as pd

# --- إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="Simplex Solver Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .main-title { color: #58a6ff; text-align: center; font-size: 3rem; font-weight: bold; }
    .step-header { background-color: #161b22; padding: 10px; border-radius: 8px; border-left: 5px solid #58a6ff; margin: 20px 0; }
    .final-card { background: linear-gradient(135deg, #238636 0%, #161b22 100%); padding: 30px; border-radius: 15px; border: 1px solid #30363d; text-align: center; }
    /* تحسين شكل الجدول */
    .styled-table { width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 1.1rem; min-width: 400px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🛡️ المحلل الاحترافي لطريقة السمبلكس</h1>", unsafe_allow_html=True)

# --- إدخال البيانات ---
with st.sidebar:
    st.header("⚙️ إعدادات المسألة")
    n_vars = st.number_input("عدد المتغيرات (X):", 2, 4, 2)
    n_const = st.number_input("عدد القيود:", 2, 4, 2)
    st.info("هذا النظام يدعم حالياً حالات Maximization مع قيود ≤")

# دالة الهدف
st.subheader("🎯 معاملات دالة الهدف (Z)")
obj_coeffs = []
cols_obj = st.columns(n_vars)
for i in range(n_vars):
    with cols_obj[i]:
        val = st.number_input(f"X{i+1}", value=0.0, step=1.0, key=f"obj_{i}")
        obj_coeffs.append(val)

# القيود
st.subheader("⛓️ القيود الهيكلية")
constraints_matrix = []
rhs_values = []
for i in range(n_const):
    st.markdown(f"**القيد رقم {i+1}**")
    cols_c = st.columns(n_vars + 1)
    row = []
    for j in range(n_vars):
        with cols_c[j]:
            val = st.number_input(f"X{j+1}", value=0.0, step=1.0, key=f"c_{i}_{j}", label_visibility="collapsed")
            row.append(val)
    with cols_c[-1]:
        rhs = st.number_input(f"RHS {i+1}", value=0.0, step=1.0, key=f"rhs_{i}", label_visibility="collapsed")
        rhs_values.append(rhs)
    constraints_matrix.append(row)

if st.button("🚀 بدأ التحليل الرياضي الشامل"):
    # 1. المرحلة التمهيدية: الصيغة القياسية
    st.markdown("<div class='step-header'><h3>1️⃣ المرحلة التمهيدية: تحويل المسألة للصيغة القياسية</h3></div>", unsafe_allow_html=True)
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    st.latex(r"Max \ Z = " + " + ".join([f"{c}X_{i+1}" for i, c in enumerate(obj_coeffs)]) + " + " + " + ".join([f"0S_{i+1}" for i in range(n_const)]))

    # 2. بدء العمليات التكرارية
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 7):
        st.markdown(f"#### 📍 جدول السمبلكس التكراري رقم ({it})")
        
        # حساب Zj و Δj (Zj - Cj)
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # بناء الجدول للعرض الاحترافي
        display_data = []
        for i in range(n_const):
            row = [basis[i], cb[i], xb[i]] + list(matrix[i])
            display_data.append(row)
        
        # إضافة صفوف Zj و Deltas
        zj_row = ["", "Zj", current_z] + list(zj)
        delta_row = ["", "Zj - Cj", ""] + list(deltas)
        
        full_cols = ["الأساس", "CB", "XB"] + col_names
        df = pd.DataFrame(display_data, columns=full_cols)
        
        # عرض صف Cj في الأعلى بشكل منفصل
        cj_df = pd.DataFrame([["", "Cj", ""] + list(cj_full)], columns=full_cols)
        
        st.write("**قيم Cj الأصلية:**")
        st.table(cj_df)
        
        st.write("**مصفوفة الحل الحالي:**")
        st.table(df.style.format(subset=["XB"] + col_names, precision=2))
        
        # عرض Zj و Deltas بتنسيق مميز
        cols_footer = st.columns(len(deltas) + 3)
        cols_footer[1].write("**Zj**")
        cols_footer[2].write(f"**{current_z:.2f}**")
        for idx, d_val in enumerate(zj):
            cols_footer[idx+3].write(f"{d_val:.2f}")
            
        cols_footer2 = st.columns(len(deltas) + 3)
        cols_footer2[1].write("**Zj - Cj**")
        for idx, d_val in enumerate(deltas):
            color = "red" if d_val < 0 else "white"
            cols_footer2[idx+3].markdown(f":{color}[{d_val:.2f}]")

        # فحص التوقف (Optimality Test)
        if np.all(deltas >= -1e-9):
            st.markdown("<div class='final-card'>", unsafe_allow_html=True)
            st.success(f"🏁 تم الوصول للحل الأمثل بنجاح في التكرار رقم {it}")
            st.subheader(f"القيمة العظمى لدالة الهدف: Z = {current_z:.2f}")
            
            final_results = []
            for var in [f"X{i+1}" for i in range(n_vars)]:
                val = xb[basis.index(var)] if var in basis else 0.0
                final_results.append(f"{var} = {val:.2f}")
            
            st.write(" | ".join(final_results))
            st.markdown("</div>", unsafe_allow_html=True)
            break
            
        # تحديد عنصر الارتكاز (Pivot)
        p_col = np.argmin(deltas)
        ratios = []
        for i in range(n_const):
            if matrix[i, p_col] > 0:
                ratios.append(xb[i] / matrix[i, p_col])
            else:
                ratios.append(np.inf)
        
        if np.all(np.isinf(ratios)):
            st.error("المسألة غير مقيدة (Unbounded Solution)")
            break
            
        p_row = np.argmin(ratios)
        
        st.write(f"➡️ المتغير الداخل: **{col_names[p_col]}** | ⬅️ المتغير الخارج: **{basis[p_row]}**")
        st.write(f"🎯 عنصر الارتكاز (Pivot): **{matrix[p_row, p_col]:.2f}**")
        st.divider()

        # تحديث المصفوفة لجدول التكرار التالي
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

    else:
        st.warning("تم الوصول للحد الأقصى من التكرارات دون إيجاد الحل الأمثل.")
