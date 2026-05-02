import streamlit as st
import numpy as np
import pandas as pd
import time

# --- 1. إعدادات الصفحة والتنسيق البصري النهائي ---
st.set_page_config(page_title="Simplex Full Analysis", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-header { font-size: 2.5rem; color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 30px; }
    
    /* تنظيف خانات الإدخال */
    button.step-up, button.step-down { display: none !important; }
    input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
    input[type=number] { -moz-appearance: textfield; text-align: center; font-size: 18px !important; }

    /* تنسيق النصوص الحسابية */
    .calculation-text { 
        font-family: 'Consolas', monospace; 
        color: #ffcc00; 
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #58a6ff;
        margin: 10px 0;
        line-height: 1.6;
    }
    .math-label { color: #58a6ff; font-weight: bold; }
    .thin-divider { border-top: 1px solid #30363d; margin: 20px 0; }
    
    /* شريط الارتكاز */
    .pivot-info { 
        background-color: #1c2128; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #ffcc00; 
        text-align: center; 
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>📑 التحليل الكامل والمفصل لطريقة السمبلكس</div>", unsafe_allow_html=True)

# --- 2. إعدادات المسألة ---
c1, c2 = st.columns(2)
with c1:
    n_vars = st.selectbox("عدد المتغيرات الأصلية (X):", [2, 3, 4], index=1)
with c2:
    n_const = st.selectbox("عدد القيود:", [2, 3, 4], index=1)

st.divider()

# --- 3. مدخلات البيانات ---
col_obj, col_const = st.columns([1, 1.3], gap="large")
with col_obj:
    st.subheader("🎯 دالة الهدف (Z)")
    obj_coeffs = [float(st.number_input(f"X{i+1}", value=0, step=1, format="%d", key=f"obj_{i}")) for i in range(n_vars)]

with col_const:
    st.subheader("⛓️ القيود (≤)")
    constraints_matrix = []
    rhs_values = []
    for i in range(n_const):
        r_cols = st.columns(n_vars + 1)
        row = [float(r_cols[j].number_input(f"L{i+1}-X{j+1}", value=0, step=1, format="%d", key=f"c_{i}_{j}", label_visibility="collapsed")) for j in range(n_vars)]
        rhs_values.append(float(r_cols[-1].number_input(f"RHS {i+1}", value=0, step=1, format="%d", key=f"rhs_{i}", label_visibility="collapsed")))
        constraints_matrix.append(row)

if st.button("🚀 بدأ التحليل التفصيلي", use_container_width=True):
    # --- الصيغة القياسية ---
    st.subheader("1️⃣ تحويل المسألة للصيغة القياسية")
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    st.markdown("**دالة الهدف الجديدة:**")
    st.code(f"Max Z = " + " + ".join([f"{int(cj_full[idx])}{col_names[idx]}" for idx in range(len(col_names))]))
    
    st.markdown("**المعادلات الموزونة:**")
    for i in range(n_const):
        eq = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {int(rhs_values[i])}"
        st.markdown(f"- القيد {i+1}: `{eq}`")
    
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

    # تهيئة المصفوفة
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    # دورات الحل
    for it in range(1, 8):
        st.subheader(f"📍 جدول السمبلكس رقم ({it})")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # عرض صف Cj
        st.table(pd.DataFrame([cj_full.astype(int)], columns=col_names, index=["Cj"]))

        # عرض الجدول الرئيسي (موحد)
        p_col_idx = np.argmin(deltas)
        main_rows = []
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            main_rows.append([basis[i], int(cb[i]), f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"])
        
        # إضافة صف Zj مباشرة للجدول
        main_rows.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        
        st.table(pd.DataFrame(main_rows, columns=["Basis", "CB", "XB"] + col_names + ["Ratio"]))

        # --- قسم التفاصيل المملة (تظهر مباشرة بدون زر) ---
        st.markdown(f"**🔍 تفاصيل حسابات الجدول {it}:**")
        
        # حساب Zj
        zj_details = ""
        for j in range(len(col_names)):
            calc = " + ".join([f"({cb[i]} × {matrix[i,j]:.2f})" for i in range(n_const)])
            zj_details += f"• **{col_names[j]}**: {calc} = **{zj[j]:.2f}**  \n"
        st.markdown(f"<div class='calculation-text'><span class='math-label'>حساب قيم Zj:</span><br>{zj_details}</div>", unsafe_allow_html=True)

        # حساب الدلتا
        delta_details = ""
        for j in range(len(col_names)):
            delta_details += f"• **{col_names[j]}**: {zj[j]:.2f} (Zj) - {cj_full[j]} (Cj) = **{deltas[j]:.2f}**  \n"
        st.markdown(f"<div class='calculation-text'><span class='math-label'>حساب صافي التقييم (Zj - Cj):</span><br>{delta_details}</div>", unsafe_allow_html=True)

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 تم الوصول للحل الأمثل بنجاح! القيمة النهائية Z = {current_z:.2f}")
            break
            
        # معلومات الارتكاز
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        st.markdown(f"""
            <div class='pivot-info'>
                📥 <b>المتغير الداخل:</b> <span class='var-tag'>{col_names[p_col_idx]}</span> | 
                🎯 <b>عنصر الارتكاز:</b> <span style='color:#3fb950; font-weight:bold;'>{matrix[p_row_idx, p_col_idx]:.2f}</span> | 
                📤 <b>المتغير الخارج:</b> <span class='var-tag'>{basis[p_row_idx]}</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

        # تحديث المصفوفة للجدول التالي
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
