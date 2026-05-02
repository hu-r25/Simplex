import streamlit as st
import numpy as np
import pandas as pd
import time

# --- التنسيق البصري ---
st.set_page_config(page_title="Simplex Interactive Coach", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .calc-step { 
        background-color: #161b22; padding: 15px; border-radius: 10px; 
        border-left: 5px solid #3fb950; margin-bottom: 10px; font-family: 'Consolas', monospace;
    }
    .highlight-num { color: #ffcc00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 مدرب السمبلكس التفاعلي")
st.info("اقتراحك تم تنفيذه: اضغط على زر 'شرح العملية الحسابية' تحت أي جدول لرؤية السحر!")

# (نفس مدخلات الكود السابق لضمان استمرارية العمل)
n_vars = st.selectbox("عدد المتغيرات (X):", [2, 3, 4], index=1)
n_const = st.selectbox("عدد القيود:", [2, 3, 4], index=1)

# ... (هنا يتم وضع كود المدخلات والمصفوفات كما في النسخة السابقة) ...
# لغرض الاختصار، سأركز على "حركة الأنميشن" داخل دورة السمبلكس:

# --- داخل دورة Loop السمبلكس ---
# (افترض أننا وصلنا لعرض الجدول)

def show_interactive_logic(it, cb, matrix, xb, zj, deltas, cj_full, col_names, basis):
    with st.expander(f"✨ شرح العملية الحسابية للجدول رقم {it} (أنميشن)"):
        st.write("---")
        with st.status(f"جاري تحليل الجدول {it}...", expanded=True) as status:
            st.write("1️⃣ نبدأ بحساب صف **Zj** (ضرب معاملات الأساس في كل عمود)...")
            time.sleep(0.5) # توقف مؤقت لمحاكاة الأنميشن
            
            for j in range(len(col_names)):
                st.markdown(f"<div class='calc-step'>عمود {col_names[j]}:<br>" + 
                            " + ".join([f"(<span class='highlight-num'>{cb[i]}</span> × {matrix[i,j]:.2f})" for i in range(len(cb))]) + 
                            f" = <span class='highlight-num'>{zj[j]:.2f}</span></div>", unsafe_allow_html=True)
                time.sleep(0.2)
            
            st.write("2️⃣ الآن نحسب **الدلتا (Zj - Cj)** لمعرفة المتغير الداخل...")
            time.sleep(0.5)
            for j in range(len(col_names)):
                st.markdown(f"🔹 {col_names[j]}: {zj[j]:.2f} - {cj_full[j]} = **{deltas[j]:.2f}**")
            
            status.update(label="✅ اكتمل الشرح الحسابي!", state="complete", expanded=False)

# (يتم استدعاء هذه الدالة تحت كل جدول في الكود الأساسي)
