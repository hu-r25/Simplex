import streamlit as st
import numpy as np
import pandas as pd
import time

# --- 1. التنسيق التعليمي والجمالي ---
st.set_page_config(page_title="Simplex Educational Coach", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-header { font-size: 2.5rem; color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 30px; }
    
    /* تنظيف خانات الإدخال */
    button.step-up, button.step-down { display: none !important; }
    input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
    input[type=number] { -moz-appearance: textfield; text-align: center; font-size: 18px !important; }

    /* تنسيق صندوق المعادلات القياسية */
    .standard-box { padding: 10px 0; margin-bottom: 20px; }
    .math-line { font-family: 'Consolas', monospace; color: #ffcc00; font-size: 1.15rem; direction: ltr; text-align: left; margin-bottom: 8px; }
    .math-label { color: #58a6ff; font-weight: bold; margin-right: 10px; direction: rtl; display: inline-block; min-width: 100px; }
    .thin-divider { border-top: 1px solid #30363d; margin: 10px 0; width: 100%; }

    /* شريط الارتكاز المنسق */
    .pivot-bar {
        display: flex; justify-content: space-around; align-items: center;
        padding: 15px; background-color: #1c2128; border-radius: 10px;
        margin: 20px 0; border-right: 5px solid #ffcc00;
    }
    .var-tag { background-color: #21262d; color: #58a6ff; padding: 2px 10px; border-radius: 5px; font-weight: bold; }
    
    /* تنسيق الجداول */
    .stTable { border: 1px solid #30363d !important; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>🎓 مدرب السمبلكس التعليمي</div>", unsafe_allow_html=True)

# --- 2. إعدادات المسألة ---
col_in1, col_in2 = st.columns(2)
with col_in1:
    n_vars = st.selectbox("عدد المتغيرات الأصلية (X):", [2, 3, 4], index=1)
with col_in2:
    n_const = st.selectbox("عدد القيود:", [2, 3, 4], index=1)

st.divider()

# --- 3. مدخلات البيانات ---
c_obj, c_const = st.columns([1, 1.3], gap="large")
with c_obj:
    st.subheader("🎯 دالة الهدف (Z)")
    obj_coeffs = [float(st.number_input(f"X{i+1}", value=0, step=1, format="%d", key=f"obj_{i}")) for i in range(n_vars)]

with c_const:
    st.subheader("⛓️ القيود (≤)")
    constraints_matrix = []
    rhs_values = []
    for i in range(n_const):
        r_cols = st.columns(n_vars + 1)
        row = [float(r_cols[j].number_input(f"L{i+1}-X{j+1}", value=0, step=1, format="%d", key=f"c_{i}_{j}", label_visibility="collapsed")) for j in range(n_vars)]
        rhs_values.append(float(r_cols[-1].number_input(f"RHS {i+1}", value=0, step=1, format="%d", key=f"rhs_{i}", label_visibility="collapsed")))
        constraints_matrix.append(row)

if st.button("🚀 بدأ التحليل واستخراج الجداول", use_container_width=True):
    # المرحلة 1: الصيغة القياسية (بدون فقاعات)
    st.subheader("1️⃣ الصيغة القياسية (Standard Form)")
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    obj_text = " + ".join([f"{int(cj_full[idx])}{col_names[idx]}" for idx in range(len(col_names))])
    
    st.markdown(f"<div class='math-line'><span class='math-label'>دالة الهدف:</span> Max Z = {obj_text}</div>", unsafe_allow_html=True)
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    
    for i in range(n_const):
        eq = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {int(rhs_values[i])}"
        st.markdown(f"<div class='math-line'><span class='math-label'>المعادلة {i+1} :</span> {eq}</div>", unsafe_allow_html=True)

    # تهيئة المصفوفة
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    # دورات الحل
    for it in range(1, 8):
        st.markdown(f"### 📍 المرحلة التكرارية رقم ({it})")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # عرض Cj
        st.table(pd.DataFrame([cj_full.astype(int)], columns=col_names, index=["Cj"]))

        # الجدول الموحد
        combined_data = []
        p_col_idx = np.argmin(deltas)
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            combined_data.append([basis[i], int(cb[i]), f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"])
        
        combined_data.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        st.table(pd.DataFrame(combined_data, columns=["Basis", "CB", "XB"] + col_names + ["Ratio"]))

        # زر الشرح التعليمي (تحت كل جدول)
        with st.expander(f"📖 شرح العمليات الحسابية للجدول {it}"):
            with st.status(f"تحليل أرقام الجدول {it}...", expanded=True) as status:
                st.write("**حساب Zj:** نضرب عمود CB في قيم الأعمدة.")
                for j in range(len(col_names)):
                    parts = [f"({cb[i]}×{matrix[i,j]:.2f})" for i in range(n_const)]
                    st.write(f"• {col_names[j]}: {' + '.join(parts)} = **{zj[j]:.2f}**")
                    time.sleep(0.05)
                
                st.write("**حساب صافي التقييم (Δj):** Zj - Cj")
                for j in range(len(col_names)):
                    st.write(f"• {col_names[j]}: {zj[j]:.2f} - {cj_full[j]} = **{deltas[j]:.2f}**")
                status.update(label="✅ تم التوضيح", state="complete", expanded=False)

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 تم الوصول للحل الأمثل: Z = {current_z:.2f}")
            break
            
        # شريط الارتكاز
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        st.markdown(f"""
            <div class="pivot-bar">
                <div>📥 <b>الداخل:</b> <span class="var-tag">{col_names[p_col_idx]}</span></div>
                <div>🎯 <b>الارتكاز:</b> <span style="background:#238636; padding:2px 8px; border-radius:5px;">{matrix[p_row_idx, p_col_idx]:.2f}</span></div>
                <div>📤 <b>الخارج:</b> <span class="var-tag">{basis[p_row_idx]}</span></div>
            </div>
        """, unsafe_allow_html=True)
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
