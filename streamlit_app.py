import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title="Belajar Klasifikasi Apel",
	page_icon=":apple:"
)

model= joblib.load("model_klasifikasi_apel.joblib")

st.title("Belajar Klasifikasi Apel")
st.markdown("Aplikasi Machine Learning Classification untuk Memprediksi Kualitas Apel")

diameter=st.slider("Diameter", 2.0, 10.10, 6.0)
berat=st.slider("Berat", 100.0, 300.0 ,150.0)
tebal_kulit=st.slider("Tebal Kulit", 0.1, 1.0, 0.5)
kadar_gula=st.slider("Kadar Gula", 9.0, 15.0, 11.0)
asal_daerah=st.pills("Asal Daerah", ["Malang", "Boyolali", "Garut"], default="Malang")
warna=st.pills("Warna", ["hijau","kuning kemerahan", "merah"], default="hijau")
musim_panen=st.pills("Musim Panen", ["hujan", "kemarau"], default="hujan")

if st.button("prediksi", type="primary"):
	data_baru= pd.DataFrame([[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, warna, musim_panen]], 
                        columns=["diameter", "berat", "tebal_kulit", "kadar_gula", "asal_daerah", "warna", "musim_panen"])
	prediksi= model.predict(data_baru)[0]
	presentase= max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi {prediksi} dengan tingkat keyakinan {presentase*100:.2f}%")
	st.balloons()

st.divider()
st.caption("Dibuat oleh **Natasya Destiana Lestari**")