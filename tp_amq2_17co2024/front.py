import streamlit as st
import requests

st.set_page_config(
    page_title="Cotizar auto",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🚗 Predictor de Precio de Autos Usados")

file_path = "assets/used-cars-dealer-meme.png"

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(file_path)

st.text("\n" * 2)
st.write("Complete los detalles del auto para obtener una predicción de su precio.")
st.text("\n")

with st.form("car_form"):
    st.subheader("Información General del Auto")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("🔹 Nombre del Auto", "Honda City 1.5 GXI")
        year = st.number_input("🔹 Año", min_value=1900, max_value=2024, value=2004)
        km_driven = st.number_input("🔹 Kilómetros Recorridos", min_value=0, value=110000)
        seats = st.number_input("🔹 Número de Asientos", min_value=1, max_value=9, value=5)
    
    with col2:
        fuel = st.selectbox("🔹 Tipo de Combustible", ["Diesel", "Petrol", "LPG", "CNG"])
        seller_type = st.selectbox("🔹 Tipo de Vendedor", ["Dealer", "Individual", "Trustmark Dealer"])
        transmission = st.selectbox("🔹 Transmisión", ["Manual", "Automatic"])
        owner = st.selectbox("🔹 Propietario", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

    st.markdown("---")
    st.subheader("Especificaciones Técnicas")

    col3, col4 = st.columns(2)
    with col3:
        mileage = st.text_input("🔹 Rendimiento (kmpl)", "12.8 kmpl")
        engine = st.text_input("🔹 Motor (CC)", "1493 CC")
    
    with col4:
        max_power = st.text_input("🔹 Potencia Máxima (bhp)", "100 bhp")
        torque = st.text_input("🔹 Torque", "113.1kgm@ 4600rpm")

    submitted = st.form_submit_button("🚀 Predecir Precio")

if submitted:
    payload = {
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "seats": seats
    }

    backend_url = "http://localhost:8800/predict/"

    try:
        response = requests.post(backend_url, json={"features": payload})
        if response.status_code == 200:
            result = response.json()
            st.success(f"💰 El precio estimado es: **₹ {result['selling_price']:.2f}**")
        else:
            st.error(f"❌ Error en la predicción: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"❌ Error de conexión: {e}")
