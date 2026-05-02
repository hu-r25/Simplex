import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات الصفحة والهوية البصرية ---
st.set_page_config(page_title="Simplex Pro Solver", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    /* تصميم الخلفية والعناصر العامة */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-header { font-size: 2.5rem; color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 30px; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
    
    /* تنسيق الحاويات والخطوات */
    .step-box { background-color: #161b22; padding: 25px; border-radius: 15px; border: 1px solid #30363d; margin: 20px 0; }
    .pivot-info { background-color: #1f2937; padding: 15px; border-left: 5px solid #ffcc00; color: #ffcc00; font-family: monospace; }
    
    /* تجميل الجداول */
    .styled-table { width: 100%; border-radius: 10px; overflow: hidden; }
    
    /* بطاقة النتيجة النهائية */
    .final-card { 
        background: linear-gradient(145deg, #161b22, #0d1117); 
        padding: 40px; border-radius: 20px; border: 2px solid #238636; 
        text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); 
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>🛡️ Simplex Solver Pro</div>", unsafe_allow_html=True)

# --- 2. مدخلات المستخدم في شريط جانبي أنيق ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.header("⚙️ الضبط الرياضي")
    n_vars = st.slider("عدد المتغيرات (X):", 2, 4, 2)
    n_const = st.slider("عدد القيود (Constraints):", 2, 4, 2)
    st.markdown("---")
    st.caption("تم التصميم لمحاكاة الحل الأكاديمي الدقيق لجامعة البصرة.")

# --- 3. تصميم واجهة الإدخال المركزية ---
col_obj, col_const = st.columns([1, 1.2], gap="large")

with col_obj:
    st.subheader("🎯 دالة الهدف (Z)")
    obj_coeffs = []
    cols = st.columns(n_vars)
    for i in range(n_vars):
        with cols[i]:
            val = st.number_input(f"معامل X{i+1}", value=0.0, step=1.0, key=f"obj_{i}")
            obj_coeffs.append(val)
    st.code(f"Max Z = " + " + ".join([f"{obj_coeffs[i]}X{i+1}" for i in range(n_vars)]))

with col_const:
    st.subheader("⛓️ القيود الهيكلية (≤)")
    constraints_matrix = []
    rhs_values = []
    for i in range(n_const):
        r_cols = st.columns(n_vars + 1)
        row = []
        for j in range(n_vars):
            with r_cols[j]:
                row.append(st.number_input(f"L{i+1}-X{j+1}", value=0.0, step=1.0, key=f"c_{i}_{j}", label_visibility="collapsed"))
        with r_cols[-1]:
            rhs_values.append(st.number_input(f"RHS {i+1}", value=0.0, step=1.0, key=f"rhs_{i}", label_visibility="collapsed"))
        constraints_matrix.append(row)

# --- 4. معالج السمبلكس والتقرير الرياضي ---
if st.button("🚀 تحليل المسألة واستخراج الجداول"):
    # تحويل للصيغة القياسية
    st.markdown("<div class='step-box'>", unsafe_allow_html=True)
    st.subheader("1️⃣ الصيغة القياسية (Standard Form)")
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    st.latex(r"Max \ Z = " + " + ".join([f"{c}X_{i+1}" for i, c in enumerate(obj_coeffs)]) + " + " + " + ".join([f"0S_{i+1}" for i in range(n_const)]))
    st.markdown("</div>", unsafe_allow_html=True)

    # تهيئة المصفوفة
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 8):
        st.write(f"### 📍 جدول السمبلكس (Iteration {it})")
        
        # حساب Zj و Zj - Cj
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # تحضير الجدول للعرض
        display_data = []
        for i in range(n_const):
            display_data.append([basis[i], cb[i], xb[i]] + list(matrix[i]))
        
        full_cols = ["الأساس (Basis)", "CB", "XB (RHS)"] + col_names
        df = pd.DataFrame(display_data, columns=full_cols)
        
        # عرض صف Cj في الأعلى
        st.markdown(f"**معاملات دالة الهدف (Cj):** `{cj_full.tolist()}`")
        
        # عرض الجدول بتنسيق أنيق
        st.table(df.style.format(precision=2).highlight_max(axis=0, color="#1f2937"))
        
        # عرض صفوف النتائج السفلية بتنسيق مميز
        c_z1, c_z2 = st.columns([1, 4])
        with c_z1: st.info(f"**Zj (Total)**")
        with c_z2: st.write(f"**Value: {current_z:.2f}** | Vector: {zj}")
        
        c_d1, c_d2 = st.columns([1, 4])
        with c_d1: st.warning(f"**Zj - Cj**")
        with c_d2: st.write(f"{deltas}")

        # اختبار الأمثلية
        if np.all(deltas >= -1e-9):
            st.markdown("<div class='final-card'>", unsafe_allow_html=True)
            st.balloons()
            st.success("🏁 تم الوصول للحل الأمثل (Optimality Reached)")
            st.markdown(f"<h2>Z = {current_z:.2f}</h2>", unsafe_allow_html=True)
            
            final_vals = []
            for var in [f"X{i+1}" for i in range(n_vars)]:
                val = xb[basis.index(var)] if var in basis else 0.0
                final_vals.append(f"<b>{var} = {val:.2f}</b>")
            st.markdown(" | ".join(final_vals), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            break
            
        # اختيار الارتكاز
        p_col = np.argmin(deltas)
        ratios = [xb[i]/matrix[i, p_col] if matrix[i, p_col] > 0 else np.inf for i in range(n_const)]
        
        if np.all(np.isinf(ratios)):
            st.error("المسألة غير مقيدة (Unbounded)")
            break
            
        p_row = np.argmin(ratios)
        
        st.markdown(f"<div class='pivot-info'>➡️ المتغير الداخل: {col_names[p_col]} | ⬅️ المتغير الخارج: {basis[p_row]} | 🎯 عنصر الارتكاز: {matrix[p_row, p_col]:.2f}</div>", unsafe_allow_html=True)
        st.divider()

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
