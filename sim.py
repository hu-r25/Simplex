import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات التصميم المتكاملة للجوال ---
st.set_page_config(page_title="Simplex Master Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
    .main-header { font-size: calc(1.3rem + 1vw); color: #58a6ff; font-weight: bold; text-align: center; margin: 20px 0; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
    
    input[type=number] { 
        text-align: center !important; background-color: #1c2128 !important; color: #ffcc00 !important;
        border: 1px solid #30363d !important; border-radius: 8px !important; height: 45px !important;
    }
    button.step-up, button.step-down { display: none !important; }

    .inequality-sign { display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: #58a6ff; height: 45px; font-weight: bold; }

    .calc-container { 
        background-color: #161b22; border: 1px solid #30363d; border-right: 5px solid #58a6ff; 
        padding: 15px; border-radius: 10px; margin: 15px 0; font-family: 'Consolas', monospace; 
        overflow-x: auto; white-space: nowrap;
    }
    .math-row { color: #d29922; direction: ltr; text-align: left; font-size: 0.95rem; margin-bottom: 8px; }
    .math-res { color: #3fb950; font-weight: bold; }
    .step-title { color: #58a6ff; font-weight: bold; margin-bottom: 10px; display: block; border-bottom: 1px solid #30363d; padding-bottom: 5px; }

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

st.markdown("<div class='main-header'>📊 المحلل الحسابي الشامل للسمبلكس</div>", unsafe_allow_html=True)

# --- 2. مدخلات المسألة ---
with st.expander("⚙️ إعدادات المسألة الأساسية", expanded=True):
    c0, c1, c2 = st.columns([1, 1, 1])
    opt_type = c0.selectbox("نوع الهدف:", ["Maximize", "Minimize"])
    n_vars = c1.selectbox("المتغيرات (X):", [2, 3, 4], index=1)
    n_const = c2.selectbox("القيود:", [2, 3, 4], index=1)

st.divider()

# دالة الهدف
st.subheader(f"🎯 دالة الهدف الأصلي ({opt_type} Z)")
obj_coeffs = []
cols_obj = st.columns(n_vars)
for i in range(n_vars):
    val = cols_obj[i].number_input(f"X{i+1}", value=0.0, step=1.0, format="%g", placeholder=f"X{i+1}", key=f"obj_{i}")
    obj_coeffs.append(val)

# القيود
st.subheader("⛓️ مصفوفة القيود")
constraints_matrix = []
rhs_values = []
for i in range(n_const):
    st.markdown(f"**📍 القيد رقم {i+1}**")
    cols_row = st.columns(list(np.ones(n_vars)) + [0.4] + [1.0])
    row = []
    for j in range(n_vars):
        v = cols_row[j].number_input(f"X{j+1}", value=0.0, step=1.0, format="%g", placeholder=f"X{j+1}", key=f"c_{i}_{j}", label_visibility="collapsed")
        row.append(v)
    cols_row[n_vars].markdown("<div class='inequality-sign'>≤</div>", unsafe_allow_html=True)
    rhs = cols_row[-1].number_input(f"الناتج", value=0.0, step=1.0, format="%g", placeholder="RHS", key=f"rhs_{i}", label_visibility="collapsed")
    constraints_matrix.append(row)
    rhs_values.append(rhs)

# --- 3. محرك الحل التفصيلي ---
if st.button("🚀 بدأ التحليل الحسابي الكامل", use_container_width=True):
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    
    # تحويل Minimize إلى Maximize
    is_min = (opt_type == "Minimize")
    cj_working = np.array(obj_coeffs + [0.0]*n_const)
    if is_min: cj_working *= -1

    # المرحلة التمهيدية
    st.markdown("### 1️⃣ الصيغة القياسية ومعالجة الإشارات")
    standard_html = "<div class='calc-container'>"
    if is_min:
        standard_html += "<span style='color:#f1c40f;'>💡 تم تحويل Minimize إلى Maximize بضرب المعاملات في (-1).</span><br>"
    
    final_matrix = []
    final_rhs = []
    for i in range(n_const):
        curr_row = np.array(constraints_matrix[i])
        curr_rhs = rhs_values[i]
        # معالجة الناتج السالب
        if curr_rhs < 0:
            standard_html += f"<span style='color:#f85149;'>⚠️ القيد {i+1} ناتجه سالب ({fmt(curr_rhs)})، تم ضربه في (-1) لضبط الحل.</span><br>"
            curr_row *= -1
            curr_rhs *= -1
        final_matrix.append(curr_row)
        final_rhs.append(curr_rhs)
        eq = " + ".join([f"{fmt(curr_row[j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {fmt(curr_rhs)}"
        standard_html += f"<p class='math-row'>C{i+1}: {eq}</p>"
    
    st.markdown(standard_html + "</div>", unsafe_allow_html=True)

    # تهيئة البيانات للجداول
    matrix = np.hstack([np.array(final_matrix), np.eye(n_const)])
    xb = np.array(final_rhs, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
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

        # العمليات الحسابية التفصيلية (الضرب والجمع والطرح)
        calc_html = "<div class='calc-container'>"
        calc_html += f"<span class='step-title'>📝 تفصيل حسابات الجدول {it}:</span>"
        calc_html += "<b>• أولاً: حساب Zj (ضرب CB في كل عمود):</b><br>"
        for j in range(len(col_names)):
            parts = [f"({fmt(cb[k])}×{fmt(matrix[k,j])})" for k in range(n_const)]
            calc_html += f"<div class='math-row'>Zj({col_names[j]}) = {' + '.join(parts)} = <span class='math-res'>{fmt(zj[j])}</span></div>"
        
        calc_html += "<br><b>• ثانياً: حساب الدلتا (Zj - Cj):</b><br>"
        for j in range(len(col_names)):
            calc_html += f"<div class='math-row'>Δ({col_names[j]}) = {fmt(zj[j])} - ({fmt(cj_working[j])}) = <span class='math-res'>{fmt(deltas[j])}</span></div>"
        st.markdown(calc_html + "</div>", unsafe_allow_html=True)

        if is_optimal:
            final_z = -current_z if is_min else current_z
            st.success(f"🏁 تم الوصول للحل الأمثل! القيمة النهائية للهدف Z = {fmt(final_z)}")
            break
            
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 1e-9 else np.inf for i in range(n_const)])
        
        st.markdown(f"""<div class='pivot-bar'><span>📥 الداخل: <span class='tag'>{col_names[p_col_idx]}</span></span><span>🎯 الارتكاز: <span class='pivot-val'>[{fmt(matrix[p_row_idx, p_col_idx])}]</span></span><span>📤 الخارج: <span class='tag'>{basis[p_row_idx]}</span></span></div>""", unsafe_allow_html=True)

        # تحديث المصفوفة
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
