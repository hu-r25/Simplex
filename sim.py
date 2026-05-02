import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات التصميم والجمالية الفائقة ---
st.set_page_config(page_title="Simplex Solver Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main-header { font-size: 2.8rem; color: #58a6ff; font-weight: bold; text-align: center; margin: 20px 0; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
    
    .calc-container { 
        background-color: #161b22; 
        border: 1px solid #30363d; 
        border-right: 6px solid #58a6ff; 
        padding: 20px; 
        border-radius: 12px; 
        margin: 20px 0; 
        font-family: 'Consolas', monospace; 
    }
    .delta-box {
        background-color: #1c2128;
        border: 1px dashed #3fb950;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .math-title { color: #58a6ff; font-weight: bold; font-size: 1.2rem; display: block; margin-bottom: 10px; }
    .math-row { color: #d29922; margin-left: 20px; direction: ltr; text-align: left; }
    .math-res { color: #3fb950; font-weight: bold; }

    .pivot-bar { 
        background: linear-gradient(90deg, #1f2937 0%, #0d1117 100%); 
        border: 1px solid #ffcc00; padding: 15px; border-radius: 10px; text-align: center; margin: 30px 0;
    }
    .tag { background-color: #21262d; color: #58a6ff; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>📈 المحلل الذكي لجدولة السمبلكس</div>", unsafe_allow_html=True)

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
if st.button("🚀 بدأ التحليل الرياضي الشامل", use_container_width=True):
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 8):
        st.markdown(f"### 📍 جدول السمبلكس التكراري رقم ({it})")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        st.table(pd.DataFrame([cj_full.astype(int)], columns=col_names, index=["Cj"]))

        p_col_idx = np.argmin(deltas)
        table_rows = []
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            table_rows.append([basis[i], int(cb[i]), f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"])
        
        table_rows.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        table_rows.append(["Δj (Zj-Cj)", "", ""] + [f"{val:.2f}" for val in deltas] + ["-"])
        st.table(pd.DataFrame(table_rows, columns=["Basis", "CB", "XB"] + col_names + ["Ratio"]))

        # --- تفاصيل حسابات الدلتا (الطلب الجديد) ---
        calc_html = "<div class='calc-container'>"
        calc_html += f"<span class='math-title'>🧪 خطوات حساب صافي التقييم (Delta) للجدول {it}:</span>"
        calc_html += "<p style='font-size:0.9rem; color:#8b949e;'>القانون: Δj = Zj - Cj (حيث Zj هو حاصل ضرب عمود CB في قيم العمود j)</p>"
        
        for j in range(len(col_names)):
            steps_zj = " + ".join([f"({cb[i]} × {matrix[i,j]:.2f})" for i in range(n_const)])
            calc_html += f"""
            <div class='delta-box'>
                <p dir='ltr' class='math-row'>1. Zj({col_names[j]}) = {steps_zj} = <span class='math-res'>{zj[j]:.2f}</span></p>
                <p dir='ltr' class='math-row'>2. Δ({col_names[j]}) = {zj[j]:.2f} (Zj) - {cj_full[j]} (Cj) = <span class='math-res'>{deltas[j]:.2f}</span></p>
            </div>
            """
        calc_html += "</div>"
        st.markdown(calc_html, unsafe_allow_html=True)

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 تم الوصول للحل الأمثل! Z = {current_z:.2f}")
            break
            
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        st.markdown(f"""
            <div class='pivot-bar'>
                📥 <b>المتغير الداخل:</b> <span class='tag'>{col_names[p_col_idx]}</span> (لأنه صاحب أقل قيمة Δ سالبة) | 
                🎯 <b>عنصر الارتكاز:</b> <span style='color:#3fb950; font-size:1.4rem; font-weight:bold;'>{matrix[p_row_idx, p_col_idx]:.2f}</span> | 
                📤 <b>المتغير الخارج:</b> <span class='tag'>{basis[p_row_idx]}</span> (لأنه صاحب أقل Ratio موجب)
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
