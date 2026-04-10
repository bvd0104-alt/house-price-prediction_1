import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.pipeline import Pipeline

# --- 1. CAU HINH TRANG ---
st.set_page_config(
    page_title="California Housing Analysis",
    layout="wide"
)

# --- 2. LOAD MO HINH ---
@st.cache_resource
def load_model():
    try:
        with open("mo_hinh_random_forest.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Loi load mo hinh: {e}")
        return None

model = load_model()

# --- 3. SIDEBAR (THANH BEN) ---
with st.sidebar:
    st.header("Thong so dau vao")
    st.markdown("---")
    
    thu_nhap = st.slider("Thu nhap (muoi ngan USD)", 0.5, 15.0, 3.5)
    tuoi_nha = st.slider("Tuoi nha trung binh", 1, 52, 20)
    
    st.subheader("Toa do")
    kinh_do = st.number_input("Kinh do", value=-122.2, format="%.2f")
    vi_do = st.number_input("Vi do", value=37.8, format="%.2f")
    
    vi_tri = st.selectbox("Vi tri so voi bien", 
                         ['gan_vinh', 'trong_dat_lien', 'gan_bien', 'dao', '<1H OCEAN'])
    
    st.markdown("---")
    st.subheader("Ha tang chi tiet")
    tong_phong = st.number_input("Tong so phong", value=1500)
    tong_ngu = st.number_input("Tong so phong ngu", value=300)
    dan_so = st.number_input("Dan so khu vuc", value=1000)
    so_ho = st.number_input("So ho gia dinh", value=400)

# --- 4. MAN HINH CHINH ---
st.title("He thong Du bao Bat dong san California")

tab1, tab2 = st.tabs(["Du doan & Ban do", "Phan tich dac trung"])

with tab1:
    col_map, col_res = st.columns([1.5, 1])
    
    with col_map:
        st.subheader("Vi tri tren ban do")
        map_df = pd.DataFrame({'lat': [vi_do], 'lon': [kinh_do]})
        st.map(map_df, zoom=10)
        
    with col_res:
        st.subheader("Ket qua du bao")
        if st.button("TINH TOAN GIA NHA", use_container_width=True):
            if so_ho == 0 or tong_phong == 0:
                st.error("Loi: So ho hoac tong phong phai lon hon 0")
            else:
                try:
                    # Tinh toan 3 cot phai sinh
                    p_t_h = tong_phong / so_ho
                    t_l_n = tong_ngu / tong_phong
                    d_t_h = dan_so / so_ho

                    # Tao DataFrame 12 cot khop voi Pipeline
                    input_df = pd.DataFrame([[
                        kinh_do, vi_do, tuoi_nha, tong_phong, tong_ngu, dan_so, so_ho, 
                        thu_nhap, p_t_h, t_l_n, d_t_h, vi_tri
                    ]], columns=[
                        'kinh_do', 'vi_do', 'tuoi_nha_trung_binh', 'tong_so_phong', 
                        'tong_so_phong_ngu', 'dan_so', 'so_ho_gia_dinh', 'thu_nhap_trung_binh',
                        'phong_tren_moi_ho', 'ty_le_phong_ngu', 'dan_so_tren_moi_ho', 'vi_tri_gan_bien'
                    ])

                    prediction = model.predict(input_df)[0]
                    
                    st.metric(label="Gia nha uoc tinh (USD)", value=f"${prediction:,.2f}")
                    
                    if prediction > 400000:
                        st.info("Phan khuc gia cao")
                    elif prediction < 100000:
                        st.info("Phan khuc gia thap")
                        
                except Exception as e:
                    st.error(f"Loi: {e}")

with tab2:
    st.subheader("Cac yeu to anh huong den gia")
    try:
        importances = model.steps[-1][1].feature_importances_
        features = [
            'Kinh do', 'Vi do', 'Tuoi nha', 'Tong phong', 'Phong ngu', 
            'Dan so', 'So ho', 'Thu nhap', 'Phong/Ho', 'Ty le ngu', 'Dan so/Ho'
        ]
        imp_df = pd.DataFrame({'Dac trung': features, 'Muc do': importances[:11]})
        imp_df = imp_df.sort_values('Muc do', ascending=True)
        
        fig = px.bar(imp_df, x='Muc do', y='Dac trung', orientation='h', 
                     color='Muc do', color_continuous_scale='Greys')
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Khong the hien thi bieu do")

st.markdown("---")
st.caption("California Housing Dataset | Random Forest Regressor")
