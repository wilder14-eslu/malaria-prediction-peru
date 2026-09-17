import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from requests.exceptions import RequestException
import os

# Configuración de página
st.set_page_config(
    page_title="Alerta Temprana de Malaria",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constantes
DEFAULT_API_URL = "https://malaria-prediction-peru.onrender.com"

# Busca el archivo en la ruta relativa desde frontend/streamlit/
DISTRICTS_FILE = os.path.join(os.path.dirname(__file__), "../../data/reference/district_names.csv")

# CSS para mejorar el UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-range {
        font-size: 14px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_districts():
    try:
        # Intenta cargar localmente
        df = pd.read_csv(DISTRICTS_FILE, dtype={"ubigeo": str})
        return df
    except FileNotFoundError:
        st.error("No se encontró el archivo de distritos local.")
        return pd.DataFrame(columns=["ubigeo", "departamento", "provincia", "distrito"])

def fetch_predictions(api_url: str, ubigeo: str):
    """Consulta la API de predicción."""
    url = f"{api_url}/predict/{ubigeo}"
    response = requests.get(url, timeout=60) # Timeout alto por el cold start de Render
    response.raise_for_status()
    return response.json()

def plot_predictions(data):
    """Genera un gráfico interactivo con Plotly."""
    horizons = []
    q50 = []
    q10 = []
    q90 = []
    
    for p in data["predictions"]:
        horizons.append(f"Semana {p['horizon']}")
        q50.append(p["quantiles"]["q0.5"])
        q10.append(p["quantiles"]["q0.1"])
        q90.append(p["quantiles"]["q0.9"])
        
    fig = go.Figure()
    
    # Área sombreada (Intervalo de confianza 80%)
    fig.add_trace(go.Scatter(
        x=horizons + horizons[::-1],
        y=q90 + q10[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Intervalo 80% (q0.1 - q0.9)'
    ))
    
    # Línea central (Mediana)
    fig.add_trace(go.Scatter(
        x=horizons,
        y=q50,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        name='Predicción (Mediana)',
        text=[f"Mediana: {val:.1f} casos" for val in q50],
        hoverinfo="text+x"
    ))
    
    fig.update_layout(
        title="Proyección a 4 semanas",
        xaxis_title="Horizonte",
        yaxis_title="Casos proyectados",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def main():
    st.title("🦟 Plataforma de Alerta Temprana de Malaria — Perú")
    st.markdown("""
    Este dashboard consume el modelo de Machine Learning que predice la cantidad de casos de malaria 
    por distrito para las próximas 1 a 4 semanas.
    """)
    
    df_distritos = load_districts()
    
    if df_distritos.empty:
        st.warning("No hay datos de distritos disponibles. Asegúrate de ejecutar el script de referencia de nombres de distrito primero.")
        return

    with st.sidebar:
        st.header("Filtros de Búsqueda")
        
        # Filtros en cascada
        dept = st.selectbox("Departamento", options=sorted(df_distritos["departamento"].unique()))
        df_prov = df_distritos[df_distritos["departamento"] == dept]
        
        prov = st.selectbox("Provincia", options=sorted(df_prov["provincia"].unique()))
        df_dist = df_prov[df_prov["provincia"] == prov]
        
        dist = st.selectbox("Distrito", options=sorted(df_dist["distrito"].unique()))
        
        ubigeo = df_dist[df_dist["distrito"] == dist]["ubigeo"].values[0]
        
        st.markdown("---")
        st.caption("Configuración Técnica")
        api_url = st.text_input("URL de la API", value=DEFAULT_API_URL)
        
    st.write(f"### Mostrando predicción para: **{dist.title()}** ({dept.title()})")
    st.caption(f"UBIGEO: {ubigeo}")
    
    if st.button("Obtener Predicciones", type="primary"):
        with st.spinner("Consultando API... (Puede tardar hasta 50s si el servidor está hibernando)"):
            try:
                data = fetch_predictions(api_url.rstrip("/"), ubigeo)
                
                # Info base del response
                epi_year = data.get("as_of_epi_year")
                epi_week = data.get("as_of_epi_week")
                if epi_year and epi_week:
                    st.info(f"📅 **Contexto temporal:** Predicción calculada desde el año epidemiológico {epi_year}, semana {epi_week}.")
                
                # Métricas superiores
                st.write("#### Resumen por semana")
                cols = st.columns(4)
                for i, p in enumerate(data["predictions"]):
                    q50 = p["quantiles"]["q0.5"]
                    q10 = p["quantiles"]["q0.1"]
                    q90 = p["quantiles"]["q0.9"]
                    
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-weight:bold;">Semana {p['horizon']}</div>
                            <div class="metric-value">{q50:.1f}</div>
                            <div class="metric-range">Casos esperados</div>
                            <div class="metric-range" style="margin-top:4px;">Rango (80%): {q10:.1f} - {q90:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.write("---")
                
                # Gráfico
                st.plotly_chart(plot_predictions(data), use_container_width=True)
                
                # Raw JSON expandible
                with st.expander("Ver respuesta cruda (JSON)"):
                    st.json(data)
                    
            except RequestException as e:
                st.error(f"Error al conectar con la API: {e}")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()
