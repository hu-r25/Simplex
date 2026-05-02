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

    /* صناديق الشرح التعليمية */
    .calculation-box { 
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #30363d; 
        margin: 15px 0;
        font-family: 'Consolas', monospace;
    }
    .math-formula { color: #ffcc00; font-size: 1.1rem; border-left: 3px solid #58a6ff; padding-left: 10px; margin: 5px 0; }

    /* شريط الارتكاز */
    .pivot-bar-container { display: flex; justify-content: space-around; align-items: center; padding: 20px 0; background-color: #1c2128; border-radius: 10px; margin: 10px 0; }
    .var-tag { background-color: #21262d; color: #58a6ff; padding: 2px 10px; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>🛡️ Simplex Solver Pro - Full Analysis</div>", unsafe_allow_html=True)

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

if st.button("🚀 بدأ التحليل الرياضي المفصل", use_container_width=True):
    # --- الصيغة القياسية ---
    st.subheader("1️⃣ الصيغة القياسية والشرح الحسابي")
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    with st.expander("📝 عرض خطوات التحويل"):
        st.write("يتم إضافة متغير مساعد (Slack Variable) لكل قيد لتحويل المتباينة إلى معادلة:")
        for i in range(n_const):
            eq = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {int(rhs_values[i])}"
            st.code(eq)

    # --- تهيئة المصفوفة ---
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 8):
        st.markdown(f"### 📍 جدول السمبلكس (التكرار رقم {it})")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # عرض Cj
        st.table(pd.DataFrame([cj_full.astype(int)], columns=col_names, index=["Cj"]))

        # بناء الجدول الموحد
        combined_data = []
        p_col_idx = np.argmin(deltas)
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            combined_data.append([basis[i], int(cb[i]), f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"])
        
        combined_data.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        combined_data.append(["Zj - Cj (Δ)", "", ""] + [f"{val:.2f}" for val in deltas] + ["-"])

        full_cols = ["Basis", "CB", "XB"] + col_names + ["Ratio"]
        st.table(pd.DataFrame(combined_data, columns=full_cols))

        # --- قسم شرح الحسابات (الذي طلبته) ---
        with st.expander(f"🔍 كيف تم حساب Zj والدلتا في الجدول {it}؟"):
            st.write("**حساب قيم Zj لكل عمود:**")
            st.write("المعادلة: $Zj = \sum (CB_i \\times Cell_{ij})$")
            for j in range(len(col_names)):
                calc_steps = " + ".join([f"({cb[i]} × {matrix[i,j]})" for i in range(n_const)])
                st.markdown(f"- عمود **{col_names[j]}**: {calc_steps} = **{zj[j]:.2f}**")
            
            st.write("**حساب قيمة Z الكلية:**")
            st.markdown(f"- $Z = \sum (CB_i \\times XB_i) = $ " + " + ".join([f"({cb[i]} × {xb[i]:.2f})" for i in range(n_const)]) + f" = **{current_z:.2f}**")

            st.write("**حساب الدلتا (Zj - Cj):**")
            for j in range(len(col_names)):
                st.markdown(f"- عمود **{col_names[j]}**: {zj[j]:.2f} - {cj_full[j]} = **{deltas[j]:.2f}**")

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 تم الوصول للحل الأمثل: Z = {current_z:.2f}")
            break
            
        # شريط الارتكاز
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        st.markdown(f"""
            <div class="pivot-bar-container">
                <div class="variable-card">📥 <b>الداخل:</b> <span class="var-tag">{col_names[p_col_idx]}</span></div>
                <div style="display:flex; align-items:center; gap:10px;">🎯 <b>الارتكاز:</b> <span style="background:#21262d; color:#3fb950; padding:4px 10px; border-radius:5px; font-weight:bold;">{matrix[p_row_idx, p_col_idx]:.2f}</span></div>
                <div class="variable-card">📤 <b>الخارج:</b> <span class="var-tag">{basis[p_row_idx]}</span></div>
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
