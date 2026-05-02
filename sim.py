import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات التصميم المتجاوب مع الجوال (Mobile Optimized) ---
st.set_page_config(page_title="Simplex Solver Mobile", layout="wide")

st.markdown("""
    <style>
    /* تحسين الرؤية على الجوال */
    .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
    
    /* جعل العناوين مرنة الحجم */
    .main-header { font-size: calc(1.5rem + 1vw); color: #58a6ff; font-weight: bold; text-align: center; margin: 15px 0; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
    
    /* حاوية الحسابات - جعلها قابلة للتمرير الأفقي في الجوال */
    .calc-container { 
        background-color: #161b22; 
        border: 1px solid #30363d; 
        border-right: 5px solid #58a6ff; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 15px 0; 
        font-family: 'Consolas', monospace; 
        line-height: 1.8;
        overflow-x: auto; /* مهم جداً للجوال */
        white-space: nowrap; /* منع انكسار المعادلات الطويلة */
    }
    
    .math-section-title { color: #58a6ff; font-weight: bold; font-size: 1.1rem; display: block; margin-bottom: 10px; border-bottom: 1px solid #30363d; }
    .math-row { color: #d29922; direction: ltr; text-align: left; font-size: 0.95rem; margin-bottom: 5px; }
    .math-res { color: #3fb950; font-weight: bold; }

    /* شريط الارتكاز المنسق للجوال */
    .pivot-bar { 
        background-color: #161b22; 
        border: 1px solid #f1c40f; 
        padding: 10px; 
        border-radius: 10px; 
        text-align: center; 
        margin: 20px 0;
        display: flex;
        flex-wrap: wrap; /* السماح بالعناصر بالنزول لسطر جديد في الجوال */
        justify-content: center;
        gap: 10px;
        font-size: 0.9rem;
    }
    .tag { background-color: #21262d; color: #58a6ff; padding: 2px 8px; border-radius: 5px; font-weight: bold; border: 1px solid #30363d; }
    .pivot-val { color: #3fb950; font-weight: bold; border: 1px solid #3fb950; padding: 0 5px; border-radius: 4px; }

    /* جعل الجداول قابلة للتمرير في الجوال */
    .stTable { 
        width: 100%; 
        border-radius: 8px; 
        overflow-x: auto !important; 
        display: block; 
        border: 1px solid #30363d !important; 
    }
    
    /* تنسيق المدخلات */
    div[data-baseweb="input"] { border-radius: 8px !important; }
    input[type=number] { text-align: center; font-size: 1rem !important; }
    
    /* تحسين الأزرار */
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>📈 محلل السمبلكس الذكي</div>", unsafe_allow_html=True)

# --- 2. مدخلات المسألة ---
st.subheader("⚙️ إعدادات المسألة")
col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    n_vars = st.selectbox("عدد المتغيرات (X):", [2, 3, 4], index=1)
with col_cfg2:
    n_const = st.selectbox("عدد القيود (≤):", [2, 3, 4], index=1)

st.divider()

# مدخلات الهدف والقيود بتنسيق مرن
st.subheader("🎯 دالة الهدف")
obj_coeffs = []
cols_obj = st.columns(n_vars)
for i in range(n_vars):
    obj_coeffs.append(float(cols_obj[i].number_input(f"X{i+1}", value=0.0, format="%.2f", key=f"obj_{i}")))

st.subheader("⛓️ القيود (≤)")
constraints_matrix = []
rhs_values = []
for i in range(n_const):
    with st.container():
        st.write(f"**القيد رقم {i+1}**")
        cols_row = st.columns(n_vars + 1)
        row = [float(cols_row[j].number_input(f"X{j+1}", value=0.0, format="%.2f", key=f"c_{i}_{j}", label_visibility="collapsed")) for j in range(n_vars)]
        rhs = float(cols_row[-1].number_input(f"RHS", value=0.0, format="%.2f", key=f"rhs_{i}", label_visibility="collapsed"))
        constraints_matrix.append(row)
        rhs_values.append(rhs)

# --- 3. محرك الحل ---
if st.button("🚀 بدأ التحليل الحسابي", use_container_width=True):
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.array(obj_coeffs + [0.0]*n_const)
    
    # 1. الصيغة القياسية
    st.markdown("### 1️⃣ الصيغة القياسية")
    standard_html = "<div class='calc-container'>"
    for i in range(n_const):
        eq = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {int(rhs_values[i])}"
        standard_html += f"<p class='math-row'>C{i+1}: {eq}</p>"
    st.markdown(standard_html + "</div>", unsafe_allow_html=True)
    
    # تهيئة المصفوفة
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    for it in range(1, 8):
        st.markdown(f"#### 📍 جدول المرحلة {it}")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # الجداول (قابلة للتمرير في الجوال)
        st.table(pd.DataFrame([cj_full], columns=col_names, index=["Cj"]))
        
        p_col_idx = np.argmin(deltas)
        table_rows = []
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            table_rows.append([basis[i], f"{cb[i]:.1f}", f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"] )
        
        table_rows.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        table_rows.append(["Δj", "", ""] + [f"{val:.2f}" for val in deltas] + ["-"])
        st.table(pd.DataFrame(table_rows, columns=["Basis", "CB", "XB"] + col_names + ["Ratio"]))

        # الحسابات التفصيلية للجوال
        calc_html = "<div class='calc-container'>"
        calc_html += f"<span class='math-section-title'>📝 حسابات الجدول {it}:</span>"
        for j in range(len(col_names)):
            calc_html += f"<p class='math-row'>• Δ({col_names[j]}) = {zj[j]:.2f}(Zj) - {cj_full[j]:.2f}(Cj) = <span class='math-res'>{deltas[j]:.2f}</span></p>"
        st.markdown(calc_html + "</div>", unsafe_allow_html=True)

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 الحل الأمثل: Z = {current_z:.2f}")
            break
            
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        
        # شريط الارتكاز المتجاوب
        st.markdown(f"""
            <div class='pivot-bar'>
                <span>📥 الداخل: <span class='tag'>{col_names[p_col_idx]}</span></span>
                <span>🎯 الارتكاز: <span class='pivot-val'>{matrix[p_row_idx, p_col_idx]:.2f}</span></span>
                <span>📤 الخارج: <span class='tag'>{basis[p_row_idx]}</span></span>
            </div>
        """, unsafe_allow_html=True)

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
