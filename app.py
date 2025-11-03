import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from datetime import datetime
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)

    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    nonzero = y_true != 0
    if np.any(nonzero):
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def naive_baseline(series, seasonal_lag=12):
    """
    Baseline estacional: pronostico = valor de hace 'seasonal_lag' meses.
    Si no hay suficientes datos, usa naive lag=1.
    """
    if len(series) > seasonal_lag:
        return series.shift(seasonal_lag)
    else:
        return series.shift(1)


# Configuración de la página
st.set_page_config(
    page_title="Lab 11",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
# st.title("Visualización de datos dinámicos")

st.markdown(
    """
    <h2 style='text-align: center; color: #764B36 ;'>Visualización de datos dinámicos</h2>
    
    """,
    

    unsafe_allow_html=True
    )

st.markdown(
    """
    <style>
    .stApp {
        # background-color: #f5f7fa;
        background: linear-gradient(180deg, #f5f7fa  0%, #EFE7DD  100%);
        color: #222222;
    }
    /* Títulos */
    h1, h2, h3 {
        color: #764B36;
        text-align: center;
    }


    </style>
    """,
    unsafe_allow_html=True
)

# Tabs
tab1, tab2 = st.tabs(["Datos", "Visualización"])

with tab1:
    # st.subheader("Tabla de Datos de Ejemplo")

    st.markdown(
    """
    <h3 style='text-align: left; color: #764B36 ;'>Tabla de Datos de Ejemplo</h3>
    """,
    unsafe_allow_html=True
    )
    
    # Generar datos de ejemplo
    df = pd.read_csv('./data/data_clean/importacion_clean_df.csv', parse_dates=['Fecha'], index_col='Fecha')
    
    st.dataframe(df, use_container_width=True)
    
    # Botón de descarga
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name="datos.csv",
        mime="text/csv"
    )

with tab2:
    # st.subheader("Visualizaciones Interactivas")

    st.markdown(
    """
    <h3 style='text-align: center; color: #764B36 ;'>Visualizaciones Interactivas</h3>
    """,
    unsafe_allow_html=True
    )
    
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
    st.sidebar.header("Filtros de Visualización")

    st.markdown(
    """
    <style>

    /* Sidebar  */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #764B36 0%, #764B36 100%);
        color: white;
        box-shadow: 2px 0 10px rgba(0,0,0,0.2);
    }

    /* revertir color a los selectbox y sus opciones */
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #000000 !important;
    }
    [data-baseweb="popover"] * {
        color: #000000 !important;
    }

    
    /* Color del texto dentro de los selectbox y inputs del sidebar */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {
        color: #222222 !important;
        background-color: white !important;
    }

    /* Color del texto dentro de las etiquetas (chips) seleccionadas */
    [data-baseweb="popover"] div {
         color: #ffffff !important;
         background-color: #D8A790 !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span {
        color: #ffffff !important;
        background-color: #D8A790 !important;
    }

    /* Títulos del sidebar siguen blancos */
     [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
         color: #FFFFFF !important;
    }
     </style>

    """,
    unsafe_allow_html=True
)

    
    st.sidebar.markdown(
    """
    <style>
    .stApp {
        background-color: #D8A790;
        color: #222222;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    
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

    st.markdown(
    """
    <h3 style='text-align: left; color: #764B36 ;'>Picos de importación anual por tipo de combustible</h3>
    """,
    unsafe_allow_html=True
    )
    # st.markdown("### ")
    
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
        st.warning("Selecciona al menos un tipo de combustible para visualizar.")
    
    st.markdown("---")
    
    # Gráfico 2: Series temporales
    # st.markdown("### Comportamiento temporal de importaciones")
    st.markdown(
    """
    <h3 style='text-align: left; color: #764B36 ;'>Comportamiento temporal de importaciones</h3>
    """,
    unsafe_allow_html=True
    )
    
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
        st.warning("Selecciona al menos un tipo de combustible para visualizar.")
    else:
        st.error("No hay datos disponibles para el rango de años seleccionado.")


    # Agregar filtros para gráficos ARIMA en el sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Gráfico 3-5: Predicciones ARIMA")

    # Filtro para año de inicio de predicción
    año_inicio_prediccion = st.sidebar.selectbox(
        "Año de inicio de predicción:",
        options=[2022, 2023, 2024],
        index=1,  # Por defecto 2023
        key="año_prediccion"
    )

    # Filtro para año final de forecast
    año_fin_forecast = st.sidebar.selectbox(
        "Año final del pronóstico:",
        options=[2024, 2025, 2026],
        index=1,  # Por defecto 2025
        key="año_fin_forecast"
    )

    # Filtro para combustibles en predicciones
    combustibles_prediccion = st.sidebar.multiselect(
        "Combustibles a predecir:",
        options=columnas_necesarias,
        default=columnas_necesarias,
        key="combustibles_prediccion"
    )

    # Checkbox para mostrar intervalo de confianza
    mostrar_intervalo = st.sidebar.checkbox(
        "Mostrar intervalo de confianza",
        value=False,
        key="intervalo_confianza"
    )

    # Sidebar - PROPHET

    st.sidebar.markdown("---")
    st.sidebar.subheader("Gráfico 7-9: Predicciones Prophet")

    horizonte_meses = st.sidebar.slider(
        "Meses a pronosticar (Prophet):",
        min_value=3, max_value=36, value=12, step=3, key="h_prophet"
    )

    seasonality_mode = st.sidebar.selectbox(
        "Modo de estacionalidad:",
        options=["additive", "multiplicative"],
        index=0, key="seasonality_mode"
    )

    cp_scale = st.sidebar.slider(
        "Changepoint prior scale:",
        min_value=0.01, max_value=0.5, value=0.1, step=0.01, key="cp_scale"
    )

    yearly_seasonality = st.sidebar.selectbox(
        "Estacionalidad anual:",
        options=[True, False, "auto"], index=0, key="yearly_seasonality"
    )

    weekly_seasonality = False
    daily_seasonality = False

    mostrar_intervalo_prophet = st.sidebar.checkbox(
        "Mostrar intervalo de confianza (Prophet)",
        value=True, key="ic_prophet"
    )

    # Sidebar - COMPARACION DE MODELOS
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comparación de Modelos")

    # Reutilizamos 'año_inicio_prediccion' y 'año_fin_forecast' del bloque ARIMA
    combustibles_comparacion = st.sidebar.multiselect(
        "Combustibles a comparar:",
        options=columnas_necesarias,
        default=columnas_necesarias,
        key="combustibles_comparacion"
    )

    metrica_objetivo = st.sidebar.selectbox(
        "Métrica para gráfico de barras:",
        options=["MAE", "RMSE", "MAPE"],
        index=0, key="metrica_bar"
    )

    # GRAFICOS 7-9: PROPHET

    for combustible in combustibles_prediccion:
        st.markdown(
            f"""
            <h3 style='text-align: left; color: #764B36;'>Predicción Prophet - {combustible}</h3>
            """,
            unsafe_allow_html=True
        )

        # Serie y preparación para Prophet 
        serie = importacion_clean_df[combustible].dropna().asfreq('MS')
        df_prophet = serie.reset_index().rename(columns={"Fecha": "ds", combustible: "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

        # Split según año_inicio_prediccion (igual a ARIMA para comparar “fair”)
        cutoff = pd.to_datetime(f"{año_inicio_prediccion}-01-01")
        train_p = df_prophet[df_prophet["ds"] < cutoff].copy()
        test_p  = df_prophet[(df_prophet["ds"] >= cutoff) &
                            (df_prophet["ds"] <= pd.to_datetime(f"{año_fin_forecast}-12-31"))].copy()

        # Entrenamiento Prophet
        m = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=cp_scale
        )
        m.fit(train_p)

        # Horizonte: para gráfica extendida usamos 'horizonte_meses'; para comparar usamos las fechas exactas de test
        future = m.make_future_dataframe(periods=horizonte_meses, freq='MS')
        fcst = m.predict(future)

        # Curvas para la UI (serie real + pronóstico)
        figp = go.Figure()
        # Real completa
        figp.add_trace(go.Scatter(
            x=df_prophet["ds"], y=df_prophet["y"],
            mode='lines', name='Serie real',
            line=dict(width=2),
            hovertemplate='<b>Fecha</b>: %{x}<br><b>Valor</b>: %{y:,.0f}<extra></extra>'
        ))
        # Pronóstico
        figp.add_trace(go.Scatter(
            x=fcst["ds"], y=fcst["yhat"],
            mode='lines', name='Pronóstico',
            line=dict(width=2, dash='dash'),
            hovertemplate='<b>Fecha</b>: %{x}<br><b>Pronóstico</b>: %{y:,.0f}<extra></extra>'
        ))

        # Intervalo de confianza (opcional)
        if mostrar_intervalo_prophet:
            figp.add_trace(go.Scatter(
                x=fcst["ds"], y=fcst["yhat_upper"],
                mode='lines', name='IC Superior',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            figp.add_trace(go.Scatter(
                x=fcst["ds"], y=fcst["yhat_lower"],
                mode='lines', name='IC Inferior',
                line=dict(width=0),
                fill='tonexty', fillcolor='rgba(31,119,180,0.15)',
                showlegend=True,
                hovertemplate='<b>IC</b>: [%{y:,.0f}]<extra></extra>'
            ))

        # Línea vertical inicio de predicción (igual que ARIMA)
        figp.add_vline(
            x=pd.to_datetime(f'{año_inicio_prediccion}-01-01').timestamp() * 1000,
            line_dash="dot", line_color="grey",
            annotation_text="Inicio de predicción", annotation_position="top"
        )

        figp.update_layout(
            title=f"Predicción Prophet para {combustible} (horizonte: {horizonte_meses} meses)",
            xaxis_title="Fecha", yaxis_title="Importaciones (Barriles)",
            hovermode='x unified', template='plotly_white',
            height=600, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        figp.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(buttons=list([
                dict(count=12, label="12m", step="month", stepmode="backward"),
                dict(count=24, label="24m", step="month", stepmode="backward"),
                dict(step="all", label="Todo")
            ]))
        )

        st.plotly_chart(figp, use_container_width=True)


    # TABLA: COMPARACIÓN DE MODELOS
    st.markdown("---")
    st.markdown(
        """
        <h3 style='text-align: left; color: #764B36;'>Comparación de modelos (ARIMA vs Prophet vs Baseline)</h3>
        """, unsafe_allow_html=True
    )

    resultados = []  # acumularemos un renglon por (combustible, modelo)

    for combustible in combustibles_comparacion:
        # --- Datos & splits coherentes con ARIMA ---
        serie = importacion_clean_df[combustible].dropna().asfreq('MS')
        train = serie[:str(año_inicio_prediccion-1)]
        test  = serie[str(año_inicio_prediccion):str(año_fin_forecast)]

        if len(test) == 0 or len(train) == 0:
            continue  # nada que comparar

        modelos_config = {
            'Diesel alto azufre': (2, 1, 5),
            'Gasolina superior': (11, 1, 6),
            'Gasolina regular': (6, 1, 12)
        }

        # --- ARIMA (reentrenamos rápido para comparación usando tus órdenes predefinidas) ---
        order = modelos_config[combustible]
        arima_model = ARIMA(train, order=order).fit()
        arima_fc = arima_model.predict(start=test.index[0], end=test.index[-1], typ='levels')
        arima_metrics = compute_metrics(test.values, arima_fc.values)
        resultados.append({
            "Combustible": combustible, "Modelo": "ARIMA",
            **arima_metrics
        })

        # --- Prophet ---
        df_p = serie.reset_index().rename(columns={"Fecha": "ds", combustible: "y"})
        df_p["ds"] = pd.to_datetime(df_p["ds"])
        cutoff = pd.to_datetime(f"{año_inicio_prediccion}-01-01")
        train_p = df_p[df_p["ds"] < cutoff].copy()
        test_p  = df_p[(df_p["ds"] >= cutoff) & (df_p["ds"] <= pd.to_datetime(f"{año_fin_forecast}-12-31"))].copy()

        m = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=cp_scale
        )
        m.fit(train_p)
        # Predecimos exactamente el rango de test
        future_test = pd.DataFrame({"ds": test_p["ds"].values})
        fc_prophet = m.predict(future_test)
        prophet_metrics = compute_metrics(test_p["y"].values, fc_prophet["yhat"].values)
        resultados.append({
            "Combustible": combustible, "Modelo": "Prophet",
            **prophet_metrics
        })

        # --- Baseline (ingenuo) ---
        bl_series = naive_baseline(serie, seasonal_lag=12)
        bl_pred = bl_series.loc[test.index]
        # Si hay NaNs (por el shift), recorta al índice disponible
        mask = (~test.isna()) & (~bl_pred.isna())
        if mask.sum() > 0:
            baseline_metrics = compute_metrics(test[mask].values, bl_pred[mask].values)
            resultados.append({
                "Combustible": combustible, "Modelo": "Baseline (Naive/Seasonal)",
                **baseline_metrics
            })

    if len(resultados) > 0:
        df_result = pd.DataFrame(resultados)
        # Ordena columnas y formatea
        df_result = df_result[["Combustible", "Modelo", "MAE", "RMSE", "MAPE"]].sort_values(
            by=["Combustible", "MAE"]
        )
        st.dataframe(df_result.style.format({
            "MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE": "{:,.2f}%"
        }), use_container_width=True)

        # Gráfico de barras por métrica seleccionada
        fig_bar = px.bar(
            df_result, x="Combustible", y=metrica_objetivo,
            color="Modelo", barmode="group",
            title=f"Comparación por {metrica_objetivo}",
            labels={metrica_objetivo: metrica_objetivo, "Combustible": "Combustible"}
        )
        fig_bar.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hay suficientes datos para comparar con los parámetros seleccionados.")


    st.sidebar.markdown("---")
    st.sidebar.subheader("Gráfico 6: Pandemia")

    # Filtro para rango de fechas de pandemia
    fecha_inicio_pandemia = st.sidebar.date_input(
        "Fecha inicio:",
        value=pd.to_datetime('2019-01-01'),
        min_value=pd.to_datetime('2019-01-01'),
        max_value=pd.to_datetime('2022-12-31'),
        key="inicio_pandemia"
    )

    fecha_fin_pandemia = st.sidebar.date_input(
        "Fecha fin:",
        value=pd.to_datetime('2021-12-31'),
        min_value=pd.to_datetime('2019-01-01'),
        max_value=pd.to_datetime('2022-12-31'),
        key="fin_pandemia"
    )

    # Filtro de combustibles para gráfico de pandemia
    combustibles_pandemia = st.sidebar.multiselect(
        "Combustibles pandemia:",
        options=columnas_necesarias,
        default=columnas_necesarias,
        key="combustibles_pandemia"
    )

    # Configuración de modelos ARIMA por combustible
    modelos_config = {
        'Diesel alto azufre': (2, 1, 5),
        'Gasolina superior': (11, 1, 6),
        'Gasolina regular': (6, 1, 12)
    }

    # Generar gráficos ARIMA para cada combustible seleccionado
    for combustible in combustibles_prediccion:
        st.markdown(
            f"""
            <h3 style='text-align: left; color: #764B36;'>Predicción ARIMA - {combustible} ({año_inicio_prediccion}–{año_fin_forecast})</h3>
            """,
            unsafe_allow_html=True
        )
        
        serie = importacion_clean_df[combustible].dropna()
        train = serie[:str(año_inicio_prediccion-1)]
        test = serie[str(año_inicio_prediccion):str(año_fin_forecast-1)]
        
        # Entrenar modelo
        order = modelos_config[combustible]
        modelo = ARIMA(train, order=order)
        resultado = modelo.fit()
        
        # Hacer predicción
        fecha_inicio = pd.to_datetime(f'{año_inicio_prediccion}-01-01')
        fecha_fin = pd.to_datetime(f'{año_fin_forecast}-12-31')
        forecast = resultado.predict(start=fecha_inicio, end=fecha_fin, typ='levels')
        
        # Crear gráfico interactivo con Plotly
        fig = go.Figure()
        
        # Serie real
        fig.add_trace(go.Scatter(
            x=serie.index,
            y=serie.values,
            mode='lines',
            name='Serie real',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Fecha</b>: %{x}<br><b>Valor</b>: %{y:,.0f}<extra></extra>'
        ))
        
        # Pronóstico
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='Pronóstico',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>Fecha</b>: %{x}<br><b>Pronóstico</b>: %{y:,.0f}<extra></extra>'
        ))
        
        # Intervalo de confianza (si está activado)
        if mostrar_intervalo:
            forecast_obj = resultado.get_forecast(steps=len(forecast))
            forecast_ci = forecast_obj.conf_int()
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast_ci.iloc[:, 1],
                mode='lines',
                name='IC Superior',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast_ci.iloc[:, 0],
                mode='lines',
                name='IC Inferior',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty',
                showlegend=True,
                hovertemplate='<b>IC</b>: [%{y:,.0f}, ' + str(forecast_ci.iloc[:, 1].values) + ']<extra></extra>'
            ))
        
        # Línea vertical para inicio de predicción
        fig.add_vline(
            x=pd.to_datetime(f'{año_inicio_prediccion}-01-01').timestamp() * 1000,
            line_dash="dot",
            line_color="grey",
            annotation_text="Inicio de predicción",
            annotation_position="top"
        )
        
        # Configuración del layout
        fig.update_layout(
            title=f"Predicción ARIMA para {combustible} ({año_inicio_prediccion}–{año_fin_forecast})",
            xaxis_title="Fecha",
            yaxis_title="Importaciones (Barriles)",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Añadir botones de zoom
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1a", step="year", stepmode="backward"),
                    dict(count=2, label="2a", step="year", stepmode="backward"),
                    dict(count=5, label="5a", step="year", stepmode="backward"),
                    dict(step="all", label="Todo")
                ])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar métricas de error si hay datos de test
        if len(test) > 0:
            forecast_test = resultado.predict(start=test.index[0], end=test.index[-1], typ='levels')
            mae = mean_absolute_error(test, forecast_test)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{mae:,.0f}")
            with col2:
                st.metric("Datos entrenamiento", len(train))
            with col3:
                st.metric("Datos prueba", len(test))

    # Gráfico 6: Comportamiento durante la pandemia
    st.markdown("---")
    st.markdown(
        """
        <h3 style='text-align: left; color: #764B36;'>Comportamiento de importaciones durante la pandemia</h3>
        """,
        unsafe_allow_html=True
    )

    if combustibles_pandemia:
        # Filtrar datos según fechas seleccionadas
        df_pandemia = importacion_clean_df.loc[
            pd.to_datetime(fecha_inicio_pandemia):pd.to_datetime(fecha_fin_pandemia)
        ]
        
        st.write(f"Mostrando datos desde {fecha_inicio_pandemia} hasta {fecha_fin_pandemia} ({len(df_pandemia)} registros)")
        
        if len(df_pandemia) > 0:
            fig6 = go.Figure()
            
            colores = {'Gasolina regular': '#1f77b4', 'Gasolina superior': '#ff7f0e', 'Diesel alto azufre': '#2ca02c'}
            
            for combustible in combustibles_pandemia:
                fig6.add_trace(go.Scatter(
                    x=df_pandemia.index,
                    y=df_pandemia[combustible],
                    mode='lines',
                    name=combustible,
                    line=dict(color=colores[combustible], width=2),
                    hovertemplate=f'<b>{combustible}</b><br>Fecha: %{{x}}<br>Valor: %{{y:,.0f}}<extra></extra>'
                ))
            
            # Línea vertical para inicio de pandemia
            fig6.add_vline(
                x=pd.to_datetime('2020-03-01').timestamp() * 1000,
                line_dash="dash",
                line_color="red",
                annotation_text="Inicio pandemia (marzo 2020)",
                annotation_position="top"
            )
            
            # Layout
            fig6.update_layout(
                title='Comportamiento de importaciones durante la pandemia',
                xaxis_title='Fecha',
                yaxis_title='Importaciones (Barriles)',
                hovermode='x unified',
                template='plotly_white',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Añadir selector de rango
            fig6.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1a", step="year", stepmode="backward"),
                        dict(step="all", label="Todo")
                    ])
                )
            )
            
            st.plotly_chart(fig6, use_container_width=True)
            
            # Análisis de impacto
            st.subheader("Análisis de impacto de la pandemia")
            
            # Calcular promedios antes y después de marzo 2020
            pre_pandemia = df_pandemia.loc[:pd.to_datetime('2020-02-29')]
            post_pandemia = df_pandemia.loc[pd.to_datetime('2020-03-01'):]
            
            if len(pre_pandemia) > 0 and len(post_pandemia) > 0:
                cols = st.columns(len(combustibles_pandemia))
                for idx, combustible in enumerate(combustibles_pandemia):
                    promedio_pre = pre_pandemia[combustible].mean()
                    promedio_post = post_pandemia[combustible].mean()
                    cambio = ((promedio_post - promedio_pre) / promedio_pre) * 100
                    
                    with cols[idx]:
                        st.metric(
                            label=combustible,
                            value=f"{promedio_post:,.0f}",
                            delta=f"{cambio:.1f}%",
                            delta_color="inverse"
                        )
        else:
            st.warning("No hay datos disponibles para el rango de fechas seleccionado.")
    else:
        st.warning("Selecciona al menos un combustible para visualizar.")

        
# Footer
st.markdown("---")
st.caption("Lab 11 - Análisis de Importaciones de Combustible")