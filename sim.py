import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات الصفحة والتنسيق الفاخر ---
st.set_page_config(page_title="Simplex Pro Solver", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-header { font-size: 2.5rem; color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 20px; }
    
    /* تنسيق الخانات بدون أزرار تحكم وجعلها نظيفة */
    button.step-up, button.step-down { display: none !important; }
    input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
    input[type=number] { -moz-appearance: textfield; text-align: center; font-size: 18px !important; }

    .step-box { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; margin: 15px 0; }
    .pivot-info { background-color: #1f2937; padding: 12px; border-left: 5px solid #ffcc00; color: #ffcc00; font-family: monospace; margin: 10px 0; }
    
    /* تنسيق الجداول الرياضية */
    .simplex-table { width: 100%; border-collapse: collapse; text-align: center; }
    .cj-header { background-color: #21262d; color: #58a6ff; font-weight: bold; }
    .final-card { background: linear-gradient(145deg, #161b22, #0d1117); padding: 30px; border-radius: 20px; border: 2px solid #238636; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>🛡️ Simplex Solver Pro</div>", unsafe_allow_html=True)

# --- 2. إعدادات المسألة ---
with st.container():
    st.subheader("⚙️ إعدادات المسألة")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        n_vars = st.selectbox("عدد المتغيرات الأصلية (X):", [2, 3, 4], index=0)
    with col_s2:
        n_const = st.selectbox("عدد القيود (Constraints):", [2, 3, 4], index=0)

st.divider()

# --- 3. مدخلات دالة الهدف والقيود (بدون بوينتات في الواجهة) ---
col_obj, col_const = st.columns([1, 1.3], gap="medium")

with col_obj:
    st.subheader("🎯 دالة الهدف (Z)")
    obj_coeffs = []
    cols = st.columns(n_vars)
    for i in range(n_vars):
        with cols[i]:
            val = st.number_input(f"X{i+1}", value=0, step=1, format="%d", key=f"obj_{i}")
            obj_coeffs.append(float(val))
    st.info(f"Max Z = " + " + ".join([f"{int(obj_coeffs[i])}X{i+1}" for i in range(n_vars)]))

with col_const:
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
    # --- المرحلة 1: الصيغة القياسية والشرح ---
    st.markdown("<div class='step-box'>", unsafe_allow_html=True)
    st.subheader("1️⃣ تحويل القيود إلى معادلات (S-Variables)")
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    for i in range(n_const):
        eq_parts = [f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)]
        eq_parts.append(f"1{s_vars[i]}")
        st.write(f"المعادلة {i+1}: {' + '.join(eq_parts)} = {int(rhs_values[i])}")
    
    full_obj = " + ".join([f"{int(cj_full[i])}{col_names[i]}" for i in range(len(cj_full))])
    st.markdown(f"**دالة الهدف الجديدة:** `Max Z = {full_obj}`")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- تهيئة المصفوفة ---
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    # --- دورات السمبلكس ---
    for it in range(1, 8):
        st.markdown(f"### 📍 جدول السمبلكس التكراري رقم ({it})")
        
        # حساب Zj و Zj - Cj
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # بناء الجدول الأكاديمي المرتب
        # صف Cj العلوي
        cj_row = ["", "", "Cj"] + [int(c) for c in cj_full] + [""]
        
        header_row = ["Basis", "CB", "XB"] + col_names + ["Ratio"]
        
        body_rows = []
        for i in range(n_const):
            # حساب Ratio
            p_col_temp = np.argmin(deltas)
            if matrix[i, p_col_temp] > 0:
                ratio = xb[i] / matrix[i, p_col_temp]
                ratio_str = f"{ratio:.2f}"
            else:
                ratio_str = "-"
            
            row = [basis[i], int(cb[i]), f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [ratio_str]
            body_rows.append(row)
        
        # دمج البيانات في DataFrame للعرض
        st.write("**توزيع معاملات Cj فوق المتغيرات:**")
        st.table(pd.DataFrame([cj_row[3:-1]], columns=col_names))
        
        df_main = pd.DataFrame(body_rows, columns=header_row)
        st.table(df_main)
        
        # عرض Zj و Zj-Cj في الأسفل بشكل مصفوفة
        f1, f2 = st.columns([1, 4])
        with f1: 
            st.markdown(f"**Zj** (Z={current_z:.2f})")
            st.markdown("**Zj - Cj**")
        with f2:
            st.write(list(np.round(zj, 2)))
            st.write(list(np.round(deltas, 2)))

        # اختبار الأمثلية
        if np.all(deltas >= -1e-9):
            st.markdown("<div class='final-card'>", unsafe_allow_html=True)
            st.success("🏁 تم الوصول للحل الأمثل")
            st.markdown(f"<h2>النتيجة النهائية: Z = {current_z:.2f}</h2>", unsafe_allow_html=True)
            
            final_res = []
            for i in range(n_vars):
                var_name = f"X{i+1}"
                val = xb[basis.index(var_name)] if var_name in basis else 0.0
                final_res.append(f"{var_name} = {val:.2f}")
            st.write(" | ".join(final_res))
            st.markdown("</div>", unsafe_allow_html=True)
            break
            
        # تحديد الارتكاز
        p_col = np.argmin(deltas)
        ratios = [xb[i]/matrix[i, p_col] if matrix[i, p_col] > 0 else np.inf for i in range(n_const)]
        
        if np.all(np.isinf(ratios)):
            st.error("Unbounded Solution!")
            break
            
        p_row = np.argmin(ratios)
        st.markdown(f"<div class='pivot-info'>➡️ المتغير الداخل: {col_names[p_col]} | ⬅️ المتغير الخارج: {basis[p_row]} | 🎯 عنصر الارتكاز: {matrix[p_row, p_col]:.2f}</div>", unsafe_allow_html=True)
        st.divider()

        # تحديث المصفوفة لجدول التكرار القادم
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
