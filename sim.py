import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات التصميم (متجاوب تماماً مع الجوال) ---
st.set_page_config(page_title="Simplex Pro Mobile", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
    .main-header { font-size: calc(1.3rem + 1vw); color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 20px; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
    
    /* تنسيق الخانات */
    input[type=number] { 
        text-align: center !important; 
        background-color: #1c2128 !important;
        color: #ffcc00 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        height: 45px !important;
    }
    button.step-up, button.step-down { display: none !important; }

    /* فاصل "أقل من أو يساوي" */
    .inequality-sign {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #58a6ff;
        height: 45px;
    }

    /* حاوية الحسابات التفصيلية */
    .calc-container { 
        background-color: #161b22; border: 1px solid #30363d; border-right: 5px solid #58a6ff; 
        padding: 15px; border-radius: 10px; margin: 15px 0; font-family: 'Consolas', monospace; 
        overflow-x: auto; white-space: nowrap;
    }
    .math-row { color: #d29922; direction: ltr; text-align: left; font-size: 0.95rem; }
    .math-res { color: #3fb950; font-weight: bold; }

    /* شريط الارتكاز (Pivot Bar) */
    .pivot-bar { 
        background-color: #161b22; border: 1px solid #f1c40f; padding: 12px; border-radius: 10px; 
        text-align: center; margin: 20px 0; display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;
    }
    .tag { background-color: #21262d; color: #58a6ff; padding: 2px 8px; border-radius: 5px; font-weight: bold; }
    .pivot-val { color: #3fb950; font-weight: bold; border: 1px solid #3fb950; padding: 0 5px; border-radius: 4px; }

    .stTable { overflow-x: auto !important; display: block; width: 100%; border: 1px solid #30363d !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>📈 محلل السمبلكس التعليمي</div>", unsafe_allow_html=True)

# --- 2. مدخلات المسألة ---
with st.expander("⚙️ إعدادات المتغيرات والقيود", expanded=True):
    c1, c2 = st.columns(2)
    n_vars = c1.selectbox("عدد متغيرات X:", [2, 3, 4], index=1)
    n_const = c2.selectbox("عدد القيود:", [2, 3, 4], index=1)

st.divider()

# مدخلات دالة الهدف
st.subheader("🎯 دالة الهدف (Z)")
obj_coeffs = []
cols_obj = st.columns(n_vars)
for i in range(n_vars):
    val = cols_obj[i].number_input(f"X{i+1}", value=0.0, placeholder=f"X{i+1}", key=f"obj_{i}")
    obj_coeffs.append(val)

# مدخلات القيود مع فاصل (≤)
st.subheader("⛓️ مصفوفة القيود")
constraints_matrix = []
rhs_values = []

for i in range(n_const):
    st.markdown(f"**📍 القيد رقم {i+1}**")
    # إنشاء أعمدة: المتغيرات + عمود للرمز + عمود للناتج
    cols_row = st.columns(list(np.ones(n_vars)) + [0.4] + [1.0])
    
    row = []
    for j in range(n_vars):
        v = cols_row[j].number_input(f"X{j+1}", value=0.0, placeholder=f"X{j+1}", key=f"c_{i}_{j}", label_visibility="collapsed")
        row.append(v)
    
    # إضافة رمز "أقل من أو يساوي" كفاصل بصري
    cols_row[n_vars].markdown("<div class='inequality-sign'>≤</div>", unsafe_allow_html=True)
    
    # الخانة الأخيرة (الناتج)
    rhs = cols_row[-1].number_input(f"الناتج", value=0.0, placeholder="الناتج", key=f"rhs_{i}", label_visibility="collapsed")
    
    constraints_matrix.append(row)
    rhs_values.append(rhs)

# --- 3. محرك الحل والعرض ---
if st.button("🚀 بدأ التحليل والشرح للجوال", use_container_width=True):
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.array(obj_coeffs + [0.0]*n_const)
    
    # تحويل القيود
    st.markdown("### 1️⃣ تحويل القيود (Standard Form)")
    standard_html = "<div class='calc-container'>"
    for i in range(n_const):
        eq = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {int(rhs_values[i])}"
        standard_html += f"<p class='math-row'>C{i+1}: {eq}</p>"
    st.markdown(standard_html + "</div>", unsafe_allow_html=True)
    
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 8):
        st.markdown(f"#### 📍 جدول المرحلة {it}")
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        st.table(pd.DataFrame([cj_full], columns=col_names, index=["Cj"]))
        p_col_idx = np.argmin(deltas)
        table_rows = []
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            table_rows.append([basis[i], f"{cb[i]:.1f}", f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"] )
        
        table_rows.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        table_rows.append(["Δj", "", ""] + [f"{val:.2f}" for val in deltas] + ["-"])
        st.table(pd.DataFrame(table_rows, columns=["الأساس", "CB", "XB"] + col_names + ["النسبة"]))

        calc_html = "<div class='calc-container'><b>• حسابات الدلتا:</b><br>"
        for j in range(len(col_names)):
            calc_html += f"<p class='math-row'>Δ({col_names[j]}) = {zj[j]:.2f}(Zj) - {cj_full[j]:.2f}(Cj) = <span class='math-res'>{deltas[j]:.2f}</span></p>"
        st.markdown(calc_html + "</div>", unsafe_allow_html=True)

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 الحل الأمثل: Z = {current_z:.2f}")
            break
            
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        
        st.markdown(f"""
            <div class='pivot-bar'>
                <span>📥 الداخل: <span class='tag'>{col_names[p_col_idx]}</span></span>
                <span>🎯 الارتكاز: <span class='pivot-val'>[{matrix[p_row_idx, p_col_idx]:.2f}]</span></span>
                <span>📤 الخارج: <span class='tag'>{basis[p_row_idx]}</span></span>
            </div>
        """, unsafe_allow_html=True)

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
        st.divider()
