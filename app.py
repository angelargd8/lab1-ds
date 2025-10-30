import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Lab 11",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("Visualización de datos dinámicos")


# Tabs
tab1, tab2 = st.tabs(["Datos", "Exploracion de datos"])

with tab1:
    st.subheader("Tabla de Datos de Importacion")
    
    # Generar datos de ejemplo
    df = pd.read_csv('./data/data_clean/importacion_clean_df.csv', parse_dates=['Fecha'], index_col='Fecha')
    
    st.dataframe(df, use_container_width=True)
    
    # Botón de descarga
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar CSV",
        data=csv,
        file_name="datos.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Visualizaciones Interactivas")
    columnas_necesarias = [
        "Gasolina regular", "Gasolina superior", 
        "Diesel alto azufre"
    ]
    importacion_clean_df = df
    
    # Picos en importaciones por año por tipo de combustibles
    for col in columnas_necesarias:
        importacion_clean_df[col] = pd.to_numeric(importacion_clean_df[col], errors="coerce")

    picos_por_anio = (
        importacion_clean_df.groupby("Año")[columnas_necesarias]
        .max()  # el pico anual
        .reset_index()
    )

    st.write("Vista previa de los datos:")
    st.dataframe(picos_por_anio.head(), use_container_width=True)

    # Crear la figura de matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for col in columnas_necesarias:
        ax.plot(picos_por_anio["Año"], picos_por_anio[col], marker='o', label=col)

    ax.set_title("Picos de importación anual por tipo de combustible")
    ax.set_xlabel("Año")
    ax.set_ylabel("Barriles (pico mensual)")
    ax.legend()
    ax.grid(True)
    
    # IMPORTANTE: Usar st.pyplot() en lugar de plt.show()
    st.pyplot(fig)

    
# Footer
st.markdown("---")
st.caption("Lab 11 - Análisis de Importaciones de Combustible")