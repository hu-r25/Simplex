import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات التصميم والجمالية الأكاديمية ---
st.set_page_config(page_title="Simplex Solver Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
    .main-header { font-size: 2.5rem; color: #58a6ff; text-align: center; margin-bottom: 20px; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
    
    .calc-container { 
        background-color: #161b22; 
        border: 1px solid #30363d; 
        border-right: 6px solid #58a6ff; 
        padding: 20px; 
        border-radius: 12px; 
        margin: 20px 0; 
        font-family: 'Consolas', monospace; 
    }
    .math-title { color: #58a6ff; font-weight: bold; font-size: 1.1rem; display: block; margin-bottom: 10px; }
    .math-row { color: #d29922; margin-left: 20px; direction: ltr; text-align: left; }
    .math-res { color: #3fb950; font-weight: bold; }
    .status-box { background-color: #0d1117; padding: 10px; border-radius: 8px; border: 1px dashed #7d8590; margin-bottom: 10px; }

    .pivot-bar { 
        background: linear-gradient(90deg, #1f2937 0%, #0d1117 100%); 
        border: 1px solid #ffcc00; padding: 15px; border-radius: 10px; text-align: center; margin: 30px 0;
    }
    .tag { background-color: #21262d; color: #58a6ff; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    .stTable { width: 100%; border-radius: 10px; overflow: hidden; border: 1px solid #30363d !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>📈 محلل جداول السمبلكس الأكاديمي</div>", unsafe_allow_html=True)

# --- 2. مدخلات المسألة ---
col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    n_vars = st.selectbox("عدد متغيرات القرار (X):", [2, 3, 4], index=1)
with col_cfg2:
    n_const = st.selectbox("عدد القيود الهيكلية:", [2, 3, 4], index=1)

st.divider()

c_obj, c_const = st.columns([1, 1.4], gap="large")
with c_obj:
    st.subheader("🎯 دالة الهدف")
    obj_coeffs = [float(st.number_input(f"معامل X{i+1}", value=0, step=1, format="%d", key=f"obj_{i}")) for i in range(n_vars)]

with c_const:
    st.subheader("⛓️ القيود (≤)")
    constraints_matrix = []
    rhs_values = []
    for i in range(n_const):
        r_cols = st.columns(n_vars + 1)
        row = [float(r_cols[j].number_input(f"L{i+1}-X{j+1}", value=0, step=1, format="%d", key=f"c_{i}_{j}", label_visibility="collapsed")) for j in range(n_vars)]
        rhs_values.append(float(r_cols[-1].number_input(f"RHS {i+1}", value=0, step=1, format="%d", key=f"rhs_{i}", label_visibility="collapsed")))
        constraints_matrix.append(row)

# --- 3. محرك الحل ---
if st.button("🚀 بدأ التحليل الرياضي الكامل", use_container_width=True):
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.array(obj_coeffs + [0.0]*n_const)
    
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 8):
        st.markdown(f"### 📍 المرحلة (الجدول) رقم {it}")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # عرض Cj العلوي
        st.table(pd.DataFrame([cj_full.astype(int)], columns=col_names, index=["Cj"]))

        p_col_idx = np.argmin(deltas)
        table_rows = []
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            table_rows.append([basis[i], f"{cb[i]:.2f}", f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"] )
        
        # إضافة صف الدلتا للجدول
        delta_row = ["Δj", "", ""] + [f"{val:.2f}" for val in deltas] + [""]
        table_rows.append(delta_row)
        
        st.table(pd.DataFrame(table_rows, columns=["Basic Variable", "CB", "XB"] + col_names + ["Min. Ratio"]))

        # --- خطوات الدلتا التفصيلية (مطابقة للصورة image_a29660.png) ---
        non_basis = [c for c in col_names if c not in basis]
        zero_vars = " = ".join(non_basis) + " = 0"
        
        calc_html = "<div class='calc-container'>"
        calc_html += f"<div class='status-box'><b>حالة المتغيرات:</b> {zero_vars}</div>"
        calc_html += f"<span class='math-title'>📝 حساب قيمة Z والدلتا (Δj):</span>"
        
        # حساب Z و Z' (كما في الصورة)
        calc_html += f"<p dir='ltr' class='math-row'>• Z' = {current_z:.2f} ⮕ Z = {-current_z:.2f}</p>"
        
        for j in range(len(col_names)):
            calc_html += f"<p dir='ltr' class='math-row'>• Δ({col_names[j]}) = {zj[j]:.2f} (Zj) - {cj_full[j]:.2f} (Cj) = <span class='math-res'>{deltas[j]:.2f}</span></p>"
        
        if np.all(deltas >= -1e-9):
            calc_html += "<p class='math-res' style='margin-top:10px;'>✅ Δj ≥ 0 : تم الوصول للحل الأمثل.</p>"
            calc_html += "</div>"
            st.markdown(calc_html, unsafe_allow_html=True)
            st.success(f"🏁 الحل النهائي: Z = {-current_z:.2f}")
            break
        
        calc_html += "</div>"
        st.markdown(calc_html, unsafe_allow_html=True)

        # تحديد الارتكاز
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        st.markdown(f"""
            <div class='pivot-bar'>
                📥 <b>المتغير الداخل:</b> <span class='tag'>{col_names[p_col_idx]}</span> (↑) | 
                🎯 <b>عنصر الارتكاز:</b> <span style='color:#3fb950; font-size:1.4rem; font-weight:bold;'>[{matrix[p_row_idx, p_col_idx]:.2f}]</span> | 
                📤 <b>المتغير الخارج:</b> <span class='tag'>{basis[p_row_idx]}</span> (↓)
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
