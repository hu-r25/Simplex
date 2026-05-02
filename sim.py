import streamlit as st
import numpy as np
import pandas as pd
import time

# --- 1. التنسيق التعليمي المتقدم ---
st.set_page_config(page_title="Simplex Educational Coach", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-header { font-size: 2.5rem; color: #58a6ff; font-weight: bold; text-align: center; margin-bottom: 30px; }
    
    /* صناديق تعليمية */
    .edu-box { 
        background-color: #161b22; padding: 20px; border-radius: 15px; 
        border: 2px solid #30363d; margin: 15px 0;
    }
    .step-badge {
        background-color: #238636; color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 0.9rem; font-weight: bold;
    }
    
    /* أنيميشن بسيط لشرح الحسابات */
    .calc-bubble {
        background: linear-gradient(90deg, #1f2937, #111827);
        border-right: 4px solid #ffcc00; padding: 15px;
        margin: 8px 0; border-radius: 8px; font-family: monospace;
    }
    .highlight { color: #ffcc00; font-weight: bold; }
    
    /* تحسين شكل الجداول */
    .stTable { border: 1px solid #30363d !important; border-radius: 10px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-header'>🎓 مدرب السمبلكس التعليمي الذكي</div>", unsafe_allow_html=True)

# --- 2. إعدادات المسألة ---
col_in1, col_in2 = st.columns(2)
with col_in1:
    n_vars = st.selectbox("كم عدد المتغيرات الأصلية عندك؟", [2, 3, 4], index=1)
with col_in2:
    n_const = st.selectbox("كم عدد القيود في المسألة؟", [2, 3, 4], index=1)

st.divider()

# --- 3. مدخلات البيانات ---
c_obj, c_const = st.columns([1, 1.3], gap="large")
with c_obj:
    st.subheader("🎯 دالة الهدف (ماذا نريد أن نحقق؟)")
    obj_coeffs = [float(st.number_input(f"معامل X{i+1}", value=0, step=1, format="%d", key=f"obj_{i}")) for i in range(n_vars)]

with c_const:
    st.subheader("⛓️ القيود (المحددات المتاحة)")
    constraints_matrix = []
    rhs_values = []
    for i in range(n_const):
        r_cols = st.columns(n_vars + 1)
        row = [float(r_cols[j].number_input(f"L{i+1}-X{j+1}", value=0, step=1, format="%d", key=f"c_{i}_{j}", label_visibility="collapsed")) for j in range(n_vars)]
        rhs_values.append(float(r_cols[-1].number_input(f"RHS {i+1}", value=0, step=1, format="%d", key=f"rhs_{i}", label_visibility="collapsed")))
        constraints_matrix.append(row)

if st.button("🚀 ابدأ الرحلة التعليمية للحل", use_container_width=True):
    # المرحلة التعليمية 1: التحويل
    st.markdown("<div class='edu-box'>", unsafe_allow_html=True)
    st.subheader("📝 الخطوة 1: موازنة المسألة (الصيغة القياسية)")
    st.write("لأننا لا نستطيع التعامل مع علامات (≤) رياضياً في الجدول، نقوم بإضافة متغيرات 'Slack' تسمى $S$ لموازنة الطرفين.")
    
    s_vars = [f"S{i+1}" for i in range(n_const)]
    col_names = [f"X{i+1}" for i in range(n_vars)] + s_vars
    cj_full = np.concatenate([obj_coeffs, [0.0]*n_const])
    
    for i in range(n_const):
        eq = " + ".join([f"{int(constraints_matrix[i][j])}X{j+1}" for j in range(n_vars)]) + f" + 1{s_vars[i]} = {int(rhs_values[i])}"
        st.markdown(f"**القيد {i+1}:** `{eq}`")
    st.markdown("</div>", unsafe_allow_html=True)

    # تهيئة المصفوفة
    matrix = np.hstack([constraints_matrix, np.eye(n_const)])
    xb = np.array(rhs_values, dtype=float)
    basis = [f"S{i+1}" for i in range(n_const)]
    cb = np.zeros(n_const)

    # دورات الحل التفاعلية
    for it in range(1, 8):
        st.markdown(f"### 📍 المرحلة التكرارية رقم ({it})")
        
        zj = np.array([np.dot(cb, matrix[:, j]) for j in range(len(cj_full))])
        deltas = zj - cj_full
        current_z = np.dot(cb, xb)

        # عرض الجدول الموحد
        combined_data = []
        p_col_idx = np.argmin(deltas)
        for i in range(n_const):
            ratio = xb[i] / matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf
            combined_data.append([basis[i], int(cb[i]), f"{xb[i]:.2f}"] + [f"{matrix[i][j]:.2f}" for j in range(len(col_names))] + [f"{ratio:.2f}" if ratio != np.inf else "-"])
        
        combined_data.append(["Zj", "", f"{current_z:.2f}"] + [f"{val:.2f}" for val in zj] + ["-"])
        
        st.table(pd.DataFrame([cj_full.astype(int)], columns=col_names, index=["Cj"]))
        st.table(pd.DataFrame(combined_data, columns=["Basis", "CB", "XB"] + col_names + ["Ratio"]))

        # --- الزر التعليمي التفاعلي (اقتراحك) ---
        with st.expander(f"📖 شرح موازنة أرقام الجدول {it}"):
            st.info("سأشرح لك الآن كيف تم توليد أرقام هذا الجدول وكأننا في قاعة دراسية:")
            
            # محاكاة أنميشن تعليمي
            with st.status("جاري شرح العمليات الحسابية...", expanded=True) as status:
                st.write("**1. حساب صف Zj (قيمة التضحية):**")
                st.write("نضرب معاملات عمود $CB$ في كل رقم داخل الأعمدة لنعرف كم سنخسر إذا أدخلنا متغيراً جديداً.")
                for j in range(len(col_names)):
                    parts = [f"({cb[i]} × {matrix[i,j]:.2f})" for i in range(n_const)]
                    st.markdown(f"<div class='calc-bubble'>عمود {col_names[j]}: {' + '.join(parts)} = <span class='highlight'>{zj[j]:.2f}</span></div>", unsafe_allow_html=True)
                    time.sleep(0.1)

                st.write("**2. حساب صافي الربح (Δj = Zj - Cj):**")
                st.write("هنا نقرر: هل إدخال هذا المتغير سيزيد أرباحنا؟ إذا كان الناتج سالباً، فهذا يعني أن المتغير مربح.")
                for j in range(len(col_names)):
                    st.markdown(f"🔸 **{col_names[j]}**: {zj[j]:.2f} (Zj) - {cj_full[j]} (Cj) = **{deltas[j]:.2f}**")
                
                status.update(label="✅ انتهى الشرح التعليمي لهذا الجدول", state="complete", expanded=False)

        if np.all(deltas >= -1e-9):
            st.balloons()
            st.success(f"🏁 مبروك! وصلنا للحل الأمثل لأن جميع قيم Δj موجبة أو صفر. القيمة النهائية Z = {current_z:.2f}")
            break
            
        # شريط الارتكاز المنسق
        p_row_idx = np.argmin([xb[i]/matrix[i, p_col_idx] if matrix[i, p_col_idx] > 0 else np.inf for i in range(n_const)])
        st.markdown(f"""
            <div style="background:#1c2128; padding:15px; border-radius:10px; display:flex; justify-content:space-around; align-items:center;">
                <div>📥 <b>الداخل (الأكثر ربحية):</b> <span style="color:#58a6ff;">{col_names[p_col_idx]}</span></div>
                <div>🎯 <b>عنصر الارتكاز:</b> <span style="background:#238636; padding:3px 8px; border-radius:5px;">{matrix[p_row_idx, p_col_idx]:.2f}</span></div>
                <div>📤 <b>الخارج (المستبدل):</b> <span style="color:#f85149;">{basis[p_row_idx]}</span></div>
            </div>
        """, unsafe_allow_html=True)
        st.divider()

        # عملية Gauss-Jordan (تحديث المصفوفة)
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
