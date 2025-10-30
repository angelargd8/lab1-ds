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
tab1, tab2 = st.tabs(["Datos", "Visualización"])

with tab1:
    st.subheader("Tabla de Datos de Ejemplo")
    
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

    # Sidebar para filtros
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtros de Visualización")
    
    # Obtener años disponibles
    años_disponibles = sorted(importacion_clean_df['Año'].unique())
    año_min = int(años_disponibles[0])
    año_max = int(años_disponibles[-1])
    
    # Filtro para gráfico de picos anuales
    st.sidebar.subheader("Gráfico 1: Picos Anuales")
    años_picos = st.sidebar.slider(
        "Rango de años:",
        min_value=año_min,
        max_value=año_max,
        value=(año_min, año_max),
        key="picos"
    )
    
    # Filtro para gráfico de series temporales
    st.sidebar.subheader("Gráfico 2: Series Temporales")
    años_series = st.sidebar.slider(
        "Rango de años:",
        min_value=año_min,
        max_value=año_max,
        value=(2019, 2021),
        key="series"
    )
    
    # Filtro de combustibles
    combustibles_seleccionados = st.sidebar.multiselect(
        "Combustibles a mostrar:",
        options=columnas_necesarias,
        default=columnas_necesarias
    )

    # Gráfico 1: Picos anuales
    st.markdown("### 📊 Picos de importación anual por tipo de combustible")
    
    # Filtrar datos según años seleccionados
    picos_filtrados = picos_por_anio[
        (picos_por_anio["Año"] >= años_picos[0]) & 
        (picos_por_anio["Año"] <= años_picos[1])
    ]
    
    st.write(f"Mostrando datos de {años_picos[0]} a {años_picos[1]}")
    st.dataframe(picos_filtrados, use_container_width=True)
    
    if combustibles_seleccionados:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        for col in combustibles_seleccionados:
            ax1.plot(picos_filtrados["Año"], picos_filtrados[col], marker='o', label=col, linewidth=2)

        ax1.set_title(f"Picos de importación anual ({años_picos[0]} - {años_picos[1]})")
        ax1.set_xlabel("Año")
        ax1.set_ylabel("Barriles (pico mensual)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        st.pyplot(fig1)
    else:
        st.warning("⚠️ Selecciona al menos un tipo de combustible para visualizar.")
    
    st.markdown("---")
    
    # Gráfico 2: Series temporales
    st.markdown("### 📈 Comportamiento temporal de importaciones")
    
    # Filtrar datos por rango de fechas
    fecha_inicio = f"{años_series[0]}-01-01"
    fecha_fin = f"{años_series[1]}-12-31"
    df_filtrado = importacion_clean_df.loc[fecha_inicio:fecha_fin]
    
    st.write(f"Mostrando datos de {años_series[0]} a {años_series[1]} ({len(df_filtrado)} registros)")
    
    if combustibles_seleccionados and len(df_filtrado) > 0:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        for col in combustibles_seleccionados:
            ax2.plot(df_filtrado.index, df_filtrado[col], label=col, linewidth=2)
        
        ax2.set_title(f'Comportamiento de importaciones ({años_series[0]} - {años_series[1]})')
        ax2.set_xlabel('Fecha')
        ax2.set_ylabel('Barriles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig2)
    elif not combustibles_seleccionados:
        st.warning("⚠️ Selecciona al menos un tipo de combustible para visualizar.")
    else:
        st.error("❌ No hay datos disponibles para el rango de años seleccionado.")

    
# Footer
st.markdown("---")
st.caption("Lab 11 - Análisis de Importaciones de Combustible")