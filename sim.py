import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات التصميم (تنسيق نظيف ومرتب للجوال) ---
st.set_page_config(page_title="Simplex Master Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
    .main-header { font-size: calc(1.3rem + 1vw); color: #58a6ff; font-weight: bold; text-align: center; margin: 20px 0; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
    
    /* تنسيق خانات الإدخال */
    input[type=number] { 
        text-align: center !important; background-color: #1c2128 !important; color: #ffcc00 !important;
        border: 1px solid #30363d !important; border-radius: 8px !important; height: 45px !important;
    }
    button.step-up, button.step-down { display: none !important; }

    .inequality-sign { display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: #58a6ff; height: 45px; font-weight: bold; }

    /* حاوية العمليات الحسابية والصيغة القياسية */
    .calc-container { 
        background-color: #161b22; border: 1px solid #30363d; border-right: 5px solid #58a6ff; 
        padding: 20px; border-radius: 12px; margin: 20px 0; font-family: 'Segoe UI', sans-serif;
        overflow-x: auto;
    }
    
    .info-msg { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; font-weight: bold; font-size: 1.05rem; }
    .bulb-icon { color: #f1c40f; font-size: 1.3rem; }
    .warn-icon { color: #f85149; font-size: 1.3rem; }
    
    .math-row { 
        color: #ffcc00; font-family: 'Consolas', monospace; font-size: 1.1rem; font-weight: bold;
        margin: 10px 0; padding-left: 25px; direction: ltr; text-align: left; white-space: nowrap;
    }

    .step-title { color: #58a6ff; font-weight: bold; margin-bottom: 10px; display: block; border-bottom: 1px solid #30363d; padding-bottom: 5px; }
    .math-res { color: #3fb950; font-weight: bold; }

    .pivot-bar { 
        background-color: #161b22; border: 1px solid #f1c40f; padding: 12px; border-radius: 10px; 
        text-align: center; margin: 20px 0; display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;
    }
    .tag { background-color: #21262d; color: #58a6ff; padding: 2px 8px; border-radius: 5px; font-weight: bold; }
    .pivot-val { color: #3fb950; font-weight: bold; border: 1px solid #3fb950; padding: 0 5px; border-radius: 4px; }
    
    .stTable { overflow-x: auto !important; display: block; width: 100%; border: 1px solid #30363d !important; }
    </style>
    """, unsafe_allow_html=True)

def fmt(num):
    if num == 0: return "0"
    if abs(num - round(num)) < 1e-9: return str(int(round(num)))
    return f"{num:.2f}".rstrip('0').rstrip('.')

st.markdown("<div class='main-header'>📊 المحلل الحسابي الدقيق للسمبلكس</div>", unsafe_allow_html=True)

# --- 2. مدخلات المسألة ---
with st.expander("⚙️ إعدادات المسألة", expanded=True):
    c0, c1, c2 = st.columns([1, 1, 1])
    opt_type = c0.selectbox("نوع الهدف:", ["Maximize", "Minimize"])
    n_vars = c1.selectbox("عدد المتغيرات (X):", [2, 3, 4], index=1)
    n_const = c2.selectbox("عدد القيود:", [2, 3, 4], index=1)

st.divider()

st.subheader(f"🎯 دالة الهدف الأصلية ({opt_type} Z)")
obj_coeffs = [st.columns(n_vars)[i].number_input(f"X{i+1}", value=0.0, format="%g", key=f"obj_{i}") for i in range(n_vars)]

st.subheader("⛓️ مصفوفة القيود")
constraints_matrix = []
rhs_values = []
for i in range(n_const):
    st.markdown(f"**📍 القيد رقم {i+1}**")
    cols_row = st.columns(list(np.ones(n_vars)) + [0.4] + [1.0])
    row = [cols_row[j].number_input(f"X{j+1}", value=0.0, key=f"c_{i}_{j}", label_visibility="collapsed") for j in range(n_vars)]
    cols_row[n_vars].markdown("<div class='inequality-sign'>≤</div>", unsafe_allow_html=True)
    rhs = cols_row[-1].number_input(f"الناتج", value=0.0, key=f"rhs_{i}", label_visibility="collapsed")
    constraints_matrix.append(row)
    rhs_values.append(rhs)

# --- 3. محرك الحل ---
if st.button("🚀 بدأ التحليل الحسابي الكامل", use_container_width=True):
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    is_min = (opt_type == "Minimize")
    
    cj_working = np.array(obj_coeffs + [0.0]*n_const)
    if is_min: cj_working *= -1

    # --- المرحلة التمهيدية: الصيغة القياسية (إصلاح التنسيق) ---
    st.markdown("### 1️⃣ المرحلة التمهيدية: الصيغة القياسية")
    
    html_output = "<div class='calc-container'>"
    
    if is_min:
        html_output += f"""
        <div class='info-msg'>
            <span class='bulb-icon'>💡</span>
            <span style='color: #f1c40f;'>بضرب المعاملات في (-1) تم تحويل Minimize إلى Maximize.</span>
        </div>
        """
    
    final_matrix = []
    final_rhs = []
    for i in range(n_const):
        curr_row = np.array(constraints_matrix[i])
        curr_rhs = rhs_values[i]
        
        if curr_rhs < 0:
            html_output += f"""
            <div class='info-msg'>
                <span class='warn-icon'>⚠️</span>
                <span style='color: #f85149;'>القيد {i+1} ناتجه سالب ({fmt(curr_rhs)})، تم ضربه في (-1) لضبط الحل.</span>
            </div>
            """
            curr_row *= -1
            curr_rhs *= -1
            
        final_matrix.append(curr_row)
        final_rhs.append(curr_rhs)
        
        eq_text = " + ".join([f"{fmt(curr_row[j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {fmt(curr_rhs)}"
        html_output += f"<div class='math-row'>C{i+1}: &nbsp; {eq_text}</div>"
    
    html_output += "</div>"
    st.markdown(html_output, unsafe_allow_html=True)

    # --- دورة الحل ---
    matrix = np.hstack([np.array(final_matrix), np.eye(n_const)])
    xb = np.array(final_rhs, dtype=float)
    basis = s_vars.copy()
    cb = np.zeros(n_const)

    for it in range(1, 11):
        st.markdown(f"#### 📍 جدول المرحلة {it}")
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_working))])
        deltas = zj - cj_working
        current_z = np.dot(cb, xb)

        st.table(pd.DataFrame([[fmt(x) for x in cj_working]], columns=col_names, index=["Cj"]))
        
        is_optimal = np.all(deltas >= -1e-9)
        p_col_idx = np.argmin(deltas)

        table_rows = []
        for i in range(n_const):
            val_p = matrix[i, p_col_idx]
            ratio = xb[i] / val_p if val_p > 1e-9 else np.inf
            table_rows.append([basis[i], fmt(cb[i]), fmt(xb[i])] + [fmt(matrix[i][j]) for j in range(len(col_names))] + [fmt(ratio) if ratio != np.inf else "-"] )
        
        table_rows.append(["Zj", "", fmt(current_z)] + [fmt(val) for val in zj] + ["-"])
        table_rows.append(["Δj", "", ""] + [fmt(val) for val in deltas] + ["-"])
        st.table(pd.DataFrame(table_rows, columns=["الأساس", "CB", "XB"] + col_names + ["النسبة"]))

        # تفصيل العمليات
        calc_box = "<div class='calc-container'><span class='step-title'>📝 تفصيل حسابات الجدول:</span>"
        for j in range(len(col_names)):
            parts = [f"({fmt(cb[k])}×{fmt(matrix[k,j])})" for k in range(n_const)]
            calc_box += f"<div class='math-row' style='color:#d29922'>Zj({col_names[j]}) = {' + '.join(parts)} = {fmt(zj[j])}</div>"
            calc_box += f"<div class='math-row' style='color:#d29922'>Δ({col_names[j]}) = {fmt(zj[j])} - ({fmt(cj_working[j])}) = <span class='math-res'>{fmt(deltas[j])}</span></div>"
        st.markdown(calc_box + "</div>", unsafe_allow_html=True)

        if is_optimal:
            final_res = -current_z if is_min else current_z
            st.success(f"🏁 تم الوصول للحل الأمثل! القيمة النهائية Z = {fmt(final_res)}")
            break
            
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 1e-9 else np.inf for i in range(n_const)])
        st.markdown(f"<div class='pivot-bar'>📥 الداخل: <span class='tag'>{col_names[p_col_idx]}</span> | 🎯 الارتكاز: <span class='pivot-val'>[{fmt(matrix[p_row_idx, p_col_idx])}]</span> | 📤 الخارج: <span class='tag'>{basis[p_row_idx]}</span></div>", unsafe_allow_html=True)

        # Gauss-Jordan
        pivot_val = matrix[p_row_idx, p_col_idx]
        matrix[p_row_idx] /= pivot_val
        xb[p_row_idx] /= pivot_val
        for i in range(n_const):
            if i != p_row_idx:
                factor = matrix[i, p_col_idx]
                matrix[i] -= factor * matrix[p_row_idx]
                xb[i] -= factor * xb[p_row_idx]
        basis[p_row_idx], cb[p_row_idx] = col_names[p_col_idx], cj_working[p_col_idx]
        st.divider()
