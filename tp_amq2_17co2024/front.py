import streamlit as st
import requests

# Título de la aplicación
st.title("🚗 Predicción de Precio de Autos Usados")
st.write("Complete los detalles del auto para obtener una predicción de su precio.")

# Imagen o separador visual
st.image("https://via.placeholder.com/800x200?text=Predicción+de+Autos", use_container_width=True)
st.markdown("---")

# Formulario para ingresar datos del auto
with st.form("car_form"):
    st.subheader("Información General del Auto")
    
    # Sección de información general
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

    # Sección de especificaciones técnicas
    col3, col4 = st.columns(2)
    with col3:
        mileage = st.text_input("🔹 Rendimiento (kmpl)", "12.8 kmpl")
        engine = st.text_input("🔹 Motor (CC)", "1493 CC")
    
    with col4:
        max_power = st.text_input("🔹 Potencia Máxima (bhp)", "100 bhp")
        torque = st.text_input("🔹 Torque", "113.1kgm@ 4600rpm")

    # Botón para enviar los datos
    submitted = st.form_submit_button("🚀 Predecir Precio")

# Si se envía el formulario
if submitted:
    # Datos a enviar al backend
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

    # URL del backend
    backend_url = "http://localhost:8000/predict/"

    # Llamada a la API FastAPI
    try:
        response = requests.post(backend_url, json={"features": payload})
        if response.status_code == 200:
            result = response.json()
            st.success(f"💰 El precio estimado es: **${result['output']:.2f}**")
        else:
            st.error(f"❌ Error en la predicción: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"❌ Error de conexión: {e}")
