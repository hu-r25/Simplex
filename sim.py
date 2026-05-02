import streamlit as st
import numpy as np
import pandas as pd

# --- 1. إعدادات التصميم المتجاوب (Mobile-First Design) ---
st.set_page_config(page_title="Simplex Mobile Coach", layout="wide")

st.markdown("""
    <style>
    /* تحسين الخطوط والألوان للجوال */
    .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
    
    /* رأس الصفحة المرن */
    .main-header { font-size: calc(1.3rem + 1vw); color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 20px; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
    
    /* حاوية الحسابات - دعم التمرير الأفقي الكامل */
    .calc-container { 
        background-color: #161b22; 
        border: 1px solid #30363d; 
        border-right: 5px solid #58a6ff; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 15px 0; 
        font-family: 'Consolas', monospace; 
        line-height: 1.8;
        overflow-x: auto; 
        -webkit-overflow-scrolling: touch;
    }
    
    /* بطاقة توضيحية (Map Card) */
    .map-card {
        background: #1c2128;
        border: 1px solid #3fb950;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 0.9rem;
    }
    .map-label { color: #3fb950; font-weight: bold; }

    .math-row { color: #d29922; direction: ltr; text-align: left; font-size: 0.95rem; white-space: nowrap; }
    .math-res { color: #3fb950; font-weight: bold; }

    /* شريط الارتكاز الذكي للجوال */
    .pivot-bar { 
        background-color: #161b22; 
        border: 1px solid #f1c40f; 
        padding: 12px; 
        border-radius: 10px; 
        text-align: center; 
        margin: 20px 0;
        display: grid;
        grid-template-columns: 1fr; /* ترتيب عمودي في الجوال */
        gap: 10px;
    }
    @media (min-width: 600px) { .pivot-bar { grid-template-columns: 1fr 1fr 1fr; } }
    
    .tag { background-color: #21262d; color: #58a6ff; padding: 4px 10px; border-radius: 5px; font-weight: bold; font-size: 0.85rem; border: 1px solid #30363d; }
    .pivot-val { color: #3fb950; font-weight: bold; border: 1px solid #3fb950; padding: 2px 8px; border-radius: 4px; }

    /* تحسين عرض الجداول على الشاشات الصغيرة */
    .stTable { overflow-x: auto !important; display: block; width: 100%; border: 1px solid #30363d !important; }
    
    /* تنسيق أزرار الإدخال */
    input[type=number] { text-align: center; font-size: 1rem !important; height: 45px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>📱 مدرب السمبلكس الحسابي</div>", unsafe_allow_html=True)

# --- 2. مدخلات المسألة ---
with st.expander("⚙️ إعدادات هيكل المسألة", expanded=True):
    c1, c2 = st.columns(2)
    n_vars = c1.selectbox("المتغيرات (X):", [2, 3, 4], index=1)
    n_const = c2.selectbox("القيود (≤):", [2, 3, 4], index=1)

st.divider()

# مدخلات الهدف والقيود
st.subheader("🎯 دالة الهدف (Z)")
obj_coeffs = []
cols_obj = st.columns(n_vars)
for i in range(n_vars):
    obj_coeffs.append(cols_obj[i].number_input(f"معامل X{i+1}", value=0.0, key=f"obj_{i}"))

st.subheader("⛓️ القيود (المحددات)")
constraints_matrix = []
rhs_values = []
for i in range(n_const):
    with st.container():
        st.markdown(f"**القيد {i+1} :**")
        cols_row = st.columns(n_vars + 1)
        row = [cols_row[j].number_input(f"X{j+1}", value=0.0, key=f"c_{i}_{j}", label_visibility="collapsed") for j in range(n_vars)]
        rhs = cols_row[-1].number_input(f"RHS", value=0.0, key=f"rhs_{i}", label_visibility="collapsed")
        constraints_matrix.append(row)
        rhs_values.append(rhs)

# --- 3. محرك الحل والعرض التوضيحي ---
if st.button("🚀 بدأ التحليل والشرح للجوال", use_container_width=True):
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.array(obj_coeffs + [0.0]*n_const)
    
    # خريطة توضيحية للقيود
    st.markdown("### 1️⃣ خريطة موازنة القيود")
    st.markdown(f"""
    <div class='map-card'>
        <span class='map-label'>📍 ماذا فعلنا؟</span><br>
        قمنا بإضافة متغيرات <b>{', '.join(s_vars)}</b> لموازنة القيود وتحويلها من متباينات (≤) إلى معادلات (=).
    </div>
    """, unsafe_allow_html=True)
    
    standard_html = "<div class='calc-container'>"
    for i in range(n_const):
        eq = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {int(rhs_values[i])}"
        standard_html += f"<p class='math-row'>المعادلة {i+1}: {eq}</p>"
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

        # عرض الجدول (يدعم السحب يميناً ويساراً في الجوال)
        st.table(pd.DataFrame([cj_full], columns=col_names, index=["Cj"]))
        
        p_col_idx = np.argmin(deltas)
        table_rows = []
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            table_rows.append([basis[i], f"{cb[i]:.1f}", f"{xb[i]:.1f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"] )
        
        table_rows.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        table_rows.append(["Δj", "", ""] + [f"{val:.2f}" for val in deltas] + ["-"])
        st.table(pd.DataFrame(table_rows, columns=["الأساس", "CB", "XB"] + col_names + ["النسبة"]))

        # تفاصيل حسابات الدلتا للجوال
        st.markdown(f"""
        <div class='map-card'>
            <span class='map-label'>🔍 توضيح أرقام الجدول {it}:</span><br>
            نحسب <b>Δj</b> لنعرف المتغير الذي سيعطينا أعلى ربح (أكبر قيمة سالبة).
        </div>
        """, unsafe_allow_html=True)
        
        calc_html = "<div class='calc-container'>"
        for j in range(len(col_names)):
            calc_html += f"<p class='math-row'>• {col_names[j]}: {zj[j]:.2f}(Zj) - {cj_full[j]:.2f}(Cj) = <span class='math-res'>{deltas[j]:.2f}</span></p>"
        st.markdown(calc_html + "</div>", unsafe_allow_html=True)

        if np.all(deltas >= -1e-9):
            st.success(f"🏁 تم الوصول للحل الأمثل: Z = {current_z:.2f}")
            break
            
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        
        # شريط الارتكاز المتجاوب
        st.markdown(f"""
            <div class='pivot-bar'>
                <div>📥 <b>الداخل:</b> <span class='tag'>{col_names[p_col_idx]}</span></div>
                <div>🎯 <b>الارتكاز:</b> <span class='pivot-val'>{matrix[p_row_idx, p_col_idx]:.2f}</span></div>
                <div>📤 <b>الخارج:</b> <span class='tag'>{basis[p_row_idx]}</span></div>
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
        st.divider()
