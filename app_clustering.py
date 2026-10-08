"""
Agrupamiento Territorial con K-Means y DBSCAN — Demostración de Streamlit + Folium
Curso: Machine Learning Aplicado al Espacio Geográfico — CNR El Salvador
Facilitadora: Jessica Martínez · doulus.jefis@gmail.com

Datos 100 % SINTÉTICOS: 1,500 parcelas en 8 departamentos de El Salvador
(4 municipios por departamento). No se usa información real del CNR.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:  # Folium es opcional: si no está instalada, la app usa un mapa plano y no se cae
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

st.set_page_config(page_title="Agrupamiento territorial · CNR", page_icon="🧭", layout="wide")
NAVY, TEAL, GOLD = "#1E2761", "#1B7F8C", "#C9A227"
st.markdown("""<style>html, body, [class*="css"] { font-size: 17px; }
h1, h2, h3 { color: #1E2761; } div[data-testid="stMetricValue"] { color: #1B7F8C; }</style>""",
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1. DATOS SINTÉTICOS: 8 departamentos × 4 municipios = 32 municipios, 1,500 parcelas
# ---------------------------------------------------------------------------
# municipio: (latitud, longitud, factor de valor, elevación base m)
TERRITORIO = {
    "Santa Ana": {"Santa Ana": (13.9942, -89.5597, 1.00, 660), "Chalchuapa": (13.9833, -89.6833, 0.80, 700),
                  "Metapán": (14.3303, -89.4472, 0.65, 620), "Coatepeque": (13.9211, -89.5453, 0.75, 750)},
    "Sonsonate": {"Sonsonate": (13.7189, -89.7242, 0.85, 220), "Izalco": (13.7406, -89.6694, 0.70, 400),
                  "Acajutla": (13.5928, -89.8275, 0.80, 10), "Nahuizalco": (13.7728, -89.7311, 0.60, 520)},
    "La Libertad": {"Santa Tecla": (13.6769, -89.2797, 1.35, 920), "Antiguo Cuscatlán": (13.6700, -89.2400, 1.80, 900),
                    "La Libertad": (13.4950, -89.3200, 1.00, 15), "Quezaltepeque": (13.8344, -89.2750, 0.75, 480)},
    "San Salvador": {"San Salvador": (13.6929, -89.2182, 1.50, 680), "Soyapango": (13.7100, -89.1400, 0.90, 650),
                     "Apopa": (13.8069, -89.1814, 0.80, 400), "Ilopango": (13.7014, -89.1103, 0.85, 620)},
    "La Paz": {"Zacatecoluca": (13.5000, -88.8700, 0.65, 120), "Santiago Nonualco": (13.5156, -88.9453, 0.60, 180),
               "San Luis La Herradura": (13.3411, -88.9406, 0.70, 5), "Olocuilta": (13.5683, -89.1156, 0.75, 250)},
    "Usulután": {"Usulután": (13.3500, -88.4500, 0.60, 80), "Jiquilisco": (13.3200, -88.5767, 0.55, 20),
                 "Santiago de María": (13.4833, -88.4667, 0.55, 900), "Berlín": (13.4950, -88.5300, 0.50, 700)},
    "San Miguel": {"San Miguel": (13.4833, -88.1833, 0.85, 110), "Chinameca": (13.5075, -88.3500, 0.55, 260),
                   "Chirilagua": (13.2397, -88.1139, 0.55, 10), "Ciudad Barrios": (13.7600, -88.2700, 0.50, 700)},
    "La Unión": {"La Unión": (13.3369, -87.8439, 0.65, 20), "Santa Rosa de Lima": (13.6256, -87.8903, 0.55, 190),
                 "Conchagua": (13.3169, -87.8650, 0.55, 150), "Anamorós": (13.7456, -87.8700, 0.45, 300)},
}
USOS = ["Residencial", "Comercial", "Industrial", "Agrícola"]
MULT_USO = {"Residencial": 1.0, "Comercial": 2.0, "Industrial": 1.25, "Agrícola": 0.28}
KM_LAT, KM_LON = 111.0, 111.0 * np.cos(np.radians(13.7))


@st.cache_data(show_spinner=False)
def generar_datos(semilla: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    munis = [(d, m, *v) for d, ms in TERRITORIO.items() for m, v in ms.items()]
    base, resto = divmod(1500, len(munis))           # 1,500 = 28 municipios × 47 + 4 × 46
    filas = []
    for i, (dep, mun, lat0, lon0, factor, elev0) in enumerate(munis):
        n = base + (1 if i < resto else 0)
        lat, lon = lat0 + rng.normal(0, 0.017, n), lon0 + rng.normal(0, 0.017, n)
        dist_centro = np.hypot((lat - lat0) * KM_LAT, (lon - lon0) * KM_LON)
        urbano = factor >= 0.85
        probs = [.55, .25, .12, .08] if urbano else [.40, .10, .05, .45]
        uso = rng.choice(USOS, n, p=probs)
        area = np.exp(rng.normal(5.9, 0.7, n)) * np.where(uso == "Agrícola", 8, 1)
        valor = (70 * factor * np.array([MULT_USO[u] for u in uso])
                 * np.exp(-0.05 * dist_centro) * np.exp(rng.normal(0, 0.18, n)))
        ndvi = np.clip(rng.normal(0.42, 0.15, n) + 0.25 * (uso == "Agrícola") - 0.2 * urbano, -0.1, 0.9)
        filas.append(pd.DataFrame({
            "departamento": dep, "municipio": mun, "latitud": lat, "longitud": lon,
            "uso_suelo": uso, "area_m2": area, "valor_m2": valor, "ndvi": ndvi,
            "elevacion_m": elev0 + rng.normal(0, 30, n) + 6 * dist_centro,
            "dist_vial_m": rng.exponential(250, n), "anomalia": ""}))
    df = pd.concat(filas, ignore_index=True)
    # Anomalías inyectadas (45 parcelas = 3 %), para ver qué detecta DBSCAN
    idx = rng.choice(len(df), 45, replace=False)
    for j, i in enumerate(idx):
        if j < 15:    # ubicación aislada
            ang, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.07, 0.11)
            df.loc[i, ["latitud", "longitud"]] += [d * np.sin(ang), d * np.cos(ang)]
            df.loc[i, "anomalia"] = "Ubicación aislada"
        elif j < 30:  # valor extremo
            df.loc[i, "valor_m2"] *= rng.choice([0.12, 4.5])
            df.loc[i, "anomalia"] = "Valor extremo"
        else:         # área extrema
            df.loc[i, "area_m2"] *= rng.uniform(15, 30)
            df.loc[i, "anomalia"] = "Área extrema"
    df["x_km"] = (df["longitud"] + 89.0) * KM_LON
    df["y_km"] = (df["latitud"] - 13.0) * KM_LAT
    df.insert(0, "id_parcela", [f"P-{i:04d}" for i in range(1, len(df) + 1)])
    return df


datos = generar_datos()

# ---------------------------------------------------------------------------
# 2. VARIABLES QUE USARÁN LOS MODELOS (tres "recetas" para comparar)
# ---------------------------------------------------------------------------
PRESETS = {
    "📍 Solo ubicación": dict(cols=["x_km", "y_km"], escalar=False, unidad="km",
                              eps=(0.5, 15.0, 3.5, 0.5), ms=5, k=8,
                              nota="Agrupa parcelas por cercanía geográfica. Distancias reales en kilómetros."),
    "💰 Ubicación + valor catastral": dict(cols=["x_km", "y_km", "log_valor"], escalar=True, unidad="desv. estándar",
                              eps=(0.1, 1.5, 0.5, 0.05), ms=5, k=6,
                              nota="Zonas donde vecinos cercanos tienen valores parecidos (zonas homogéneas de valor)."),
    "🌿 Atributos del terreno": dict(cols=["log_valor", "log_area", "ndvi", "elevacion_m"],
                              escalar=True, unidad="desv. estándar", eps=(0.3, 2.5, 0.9, 0.1), ms=6, k=3,
                              nota="Agrupa por parecido de características, sin importar dónde estén en el mapa."),
}


def matriz(preset: str):
    d = datos.assign(log_area=np.log(datos["area_m2"]), log_valor=np.log(datos["valor_m2"]))
    X = d[PRESETS[preset]["cols"]].to_numpy(float)
    return StandardScaler().fit_transform(X) if PRESETS[preset]["escalar"] else X


@st.cache_data(show_spinner=False)
def curva_codo(preset: str):
    X, filas = matriz(preset), []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        filas.append((k, km.inertia_, silhouette_score(X, km.labels_)))
    return pd.DataFrame(filas, columns=["k", "inercia", "silhouette"])


@st.cache_data(show_spinner=False)
def correr_kmeans(preset: str, k: int):
    X = matriz(preset)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    return km.labels_, km.cluster_centers_, km.inertia_, silhouette_score(X, km.labels_), davies_bouldin_score(X, km.labels_)


@st.cache_data(show_spinner=False)
def correr_dbscan(preset: str, eps: float, min_samples: int):
    X = matriz(preset)
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_


@st.cache_data(show_spinner=False)
def curva_k_distancia(preset: str, k: int):
    d, _ = NearestNeighbors(n_neighbors=k).fit(matriz(preset)).kneighbors(matriz(preset))
    return np.sort(d[:, -1])


def metricas_dbscan(preset, etiquetas):
    X = matriz(preset)
    mask = etiquetas != -1
    n_cl = len(set(etiquetas[mask]))
    sil = silhouette_score(X[mask], etiquetas[mask]) if n_cl >= 2 and mask.sum() > n_cl else np.nan
    return n_cl, int((~mask).sum()), sil


# ---------------------------------------------------------------------------
# 3. MAPAS (Folium)
# ---------------------------------------------------------------------------
CONTORNO_SV = [[-90.0983,13.7314],[-90.1143,13.7963],[-90.107,13.8441],[-90.0871,13.8705],[-90.0474,13.8943],[-90.023,13.937],[-89.8908,14.0359],[-89.8359,14.0591],[-89.803,14.0555],[-89.7623,14.03],[-89.7476,14.0376],[-89.7526,14.0752],[-89.7099,14.1487],[-89.6898,14.17],[-89.6383,14.2006],[-89.5309,14.2256],[-89.5245,14.2318],[-89.5247,14.2474],[-89.5449,14.2618],[-89.5586,14.3028],[-89.5922,14.3138],[-89.5984,14.328],[-89.5761,14.3531],[-89.5906,14.386],[-89.5841,14.4018],[-89.5699,14.412],[-89.5552,14.4108],[-89.5442,14.4001],[-89.5414,14.3815],[-89.503,14.4201],[-89.4853,14.4274],[-89.4468,14.4155],[-89.4311,14.4187],[-89.3982,14.4454],[-89.3969,14.426],[-89.3906,14.436],[-89.3618,14.4155],[-89.3076,14.4071],[-89.2765,14.3927],[-89.2255,14.3821],[-89.2002,14.3647],[-89.154,14.3513],[-89.1222,14.3947],[-89.1075,14.396],[-89.0918,14.3748],[-89.0945,14.3449],[-89.0831,14.3353],[-89.033,14.3237],[-89.0181,14.2717],[-88.9835,14.2421],[-88.966,14.1914],[-88.9572,14.1852],[-88.9216,14.1999],[-88.906,14.1996],[-88.8813,14.1842],[-88.8603,14.1585],[-88.8388,14.1004],[-88.7773,14.0997],[-88.7464,14.1113],[-88.7392,14.0999],[-88.7466,14.0525],[-88.731,14.0425],[-88.7049,14.038],[-88.6927,14.0253],[-88.6909,14.01],[-88.6339,14.0145],[-88.5581,13.9909],[-88.5044,13.9837],[-88.4954,13.9751],[-88.5023,13.9689],[-88.4961,13.9557],[-88.5023,13.9171],[-88.4887,13.8813],[-88.488,13.8653],[-88.4968,13.8512],[-88.4702,13.8525],[-88.4332,13.8711],[-88.3929,13.8798],[-88.3601,13.8724],[-88.3551,13.891],[-88.3262,13.8851],[-88.3193,13.8976],[-88.2854,13.906],[-88.2636,13.9331],[-88.2364,13.9381],[-88.2316,13.9518],[-88.2444,13.9661],[-88.2449,13.9893],[-88.2351,13.9929],[-88.1775,13.9852],[-88.1254,13.9913],[-88.099,13.9898],[-88.0875,13.9805],[-88.0729,13.9434],[-88.0236,13.8912],[-88.0153,13.8664],[-87.9548,13.8925],[-87.9253,13.8796],[-87.9001,13.8941],[-87.8609,13.8902],[-87.8329,13.9153],[-87.8179,13.9159],[-87.8085,13.9086],[-87.7906,13.8705],[-87.7036,13.815],[-87.7378,13.7388],[-87.7314,13.7222],[-87.755,13.6899],[-87.7601,13.6169],[-87.7899,13.5328],[-87.7795,13.51],[-87.7692,13.5064],[-87.748,13.5132],[-87.7334,13.5077],[-87.7206,13.487],[-87.7212,13.4608],[-87.7384,13.4417],[-87.8172,13.4066],[-87.8257,13.4113],[-87.8383,13.4409],[-87.8664,13.3933],[-87.8738,13.3653],[-87.8436,13.3485],[-87.7915,13.305],[-87.7899,13.287],[-87.7981,13.2657],[-87.818,13.2504],[-87.8923,13.2139],[-87.9125,13.1982],[-87.916,13.1822],[-87.8929,13.1666],[-87.9282,13.1586],[-88.0984,13.174],[-88.2083,13.1604],[-88.3009,13.1742],[-88.3249,13.1666],[-88.3523,13.174],[-88.3523,13.1802],[-88.3297,13.1817],[-88.3281,13.1971],[-88.3402,13.2161],[-88.3584,13.2287],[-88.3379,13.1945],[-88.3918,13.1939],[-88.4069,13.1877],[-88.3972,13.1836],[-88.3659,13.1877],[-88.3659,13.1802],[-88.3892,13.1708],[-88.4148,13.1716],[-88.4348,13.1837],[-88.441,13.2082],[-88.4274,13.2014],[-88.4137,13.2212],[-88.4308,13.2288],[-88.4496,13.2546],[-88.4621,13.2622],[-88.4502,13.2254],[-88.4547,13.215],[-88.5298,13.2429],[-88.554,13.2764],[-88.5987,13.2839],[-88.7228,13.2696],[-88.7087,13.2605],[-88.6666,13.2558],[-88.5774,13.2628],[-88.5366,13.2287],[-88.5366,13.2212],[-88.5987,13.2355],[-88.5987,13.2287],[-88.4739,13.1986],[-88.4547,13.1802],[-88.4617,13.1683],[-88.4796,13.1712],[-88.5275,13.1984],[-88.7857,13.2453],[-89.0648,13.372],[-89.1835,13.4431],[-89.2515,13.4723],[-89.3249,13.4893],[-89.385,13.4953],[-89.5271,13.4942],[-89.6798,13.5347],[-89.7131,13.528],[-89.808,13.5269],[-89.8233,13.5371],[-89.8371,13.5968],[-89.8497,13.6069],[-89.9513,13.6641],[-90.0983,13.7314]]
PALETA = ["#1B7F8C", "#C9A227", "#D64545", "#2E9E5B", "#6B4FA3", "#E8833A", "#3B6FB6", "#B5517F", "#7A8B99", "#5BB5A2",
          "#8C564B", "#17BECF", "#BCBD22", "#E377C2", "#393B79", "#637939"]
COLOR_RUIDO = "#111827"


def color_cluster(c):
    return COLOR_RUIDO if c == -1 else PALETA[int(c) % len(PALETA)]


def nombre_cluster(c):
    return "Ruido / anomalía" if c == -1 else f"Grupo {int(c) + 1}"


def mapa_folium(df, centros=None, alto=560):
    m = folium.Map(location=[13.78, -88.9], zoom_start=9, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="Calles (OpenStreetMap)").add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, HERE, Garmin", name="Gris claro (Esri)", show=False).add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, Maxar, Earthstar Geographics", name="Imagen satelital (Esri)", show=False).add_to(m)
    folium.GeoJson({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [CONTORNO_SV]}},
                   name="Contorno de El Salvador",
                   style_function=lambda _: {"color": NAVY, "weight": 2.5, "fillColor": TEAL, "fillOpacity": 0.05}).add_to(m)

    capa = folium.FeatureGroup(name="Parcelas")
    for f in df.to_dict("records"):
        c, ruido = color_cluster(f["cluster"]), f["cluster"] == -1
        tip = (f"<b>{f['id_parcela']}</b><br>{f['municipio']}, {f['departamento']}<br>"
               f"Valor: US$ {f['valor_m2']:,.0f}/m²<br>Área: {f['area_m2']:,.0f} m²<br>"
               f"<b>{nombre_cluster(f['cluster'])}</b>")
        folium.CircleMarker([f["latitud"], f["longitud"]], radius=7 if ruido else 5, color=c,
                            weight=2 if ruido else 0.6, fill=True, fill_color=c,
                            fill_opacity=0.35 if ruido else 0.85,
                            tooltip=folium.Tooltip(tip, sticky=True)).add_to(capa)
    capa.add_to(m)

    if centros is not None:  # centroides de K-Means (estrella)
        cc = folium.FeatureGroup(name="Centroides K-Means")
        for i, (lat, lon) in enumerate(centros):
            folium.Marker([lat, lon], tooltip=f"Centroide del {nombre_cluster(i)}", icon=folium.DivIcon(html=(
                f'<div style="font-size:26px;line-height:26px;color:{color_cluster(i)};'
                'text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;'
                'transform:translate(-50%,-50%)">★</div>'))).add_to(cc)
        cc.add_to(m)

    deps = folium.FeatureGroup(name="Nombres de departamento")
    for dep, ms in TERRITORIO.items():
        lat = np.mean([v[0] for v in ms.values()]); lon = np.mean([v[1] for v in ms.values()])
        folium.Marker([lat + 0.13, lon], icon=folium.DivIcon(html=(
            f'<div style="font:700 11px sans-serif;color:{NAVY};text-shadow:0 0 3px #fff,0 0 3px #fff;'
            f'white-space:nowrap;transform:translateX(-50%)">{dep.upper()}</div>'))).add_to(deps)
    deps.add_to(m)

    etiquetas = sorted(df["cluster"].unique())
    items = "".join(f'<div><span style="background:{color_cluster(c)};display:inline-block;width:12px;height:12px;'
                    f'border-radius:50%;margin-right:6px"></span>{nombre_cluster(c)} ({(df["cluster"] == c).sum()})</div>'
                    for c in etiquetas)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:white;padding:8px 12px;'
        'border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:13px sans-serif;color:#1E2761;'
        f'max-height:260px;overflow:auto">{items}</div>'))
    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, height=alto, use_container_width=True, returned_objects=[])


def mapa_plano(df, centros=None, alto=560):
    d = df.assign(grupo=df["cluster"].map(nombre_cluster))
    orden = [nombre_cluster(c) for c in sorted(df["cluster"].unique())]
    fig = px.scatter(d, x="longitud", y="latitud", color="grupo", height=alto, hover_name="id_parcela",
                     hover_data=["municipio", "valor_m2"], category_orders={"grupo": orden},
                     color_discrete_map={nombre_cluster(c): color_cluster(c) for c in df["cluster"].unique()})
    lon, lat = zip(*CONTORNO_SV)
    fig.add_scatter(x=lon, y=lat, mode="lines", line=dict(color=NAVY, width=2), hoverinfo="skip", showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(np.radians(13.6)))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#F4F7FB", legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def mostrar_mapa(df, centros=None):
    (mapa_folium if usar_folium else mapa_plano)(df, centros)


def perfil(df):
    g = df.assign(grupo=df["cluster"].map(nombre_cluster)).groupby("grupo")
    t = g.agg(parcelas=("id_parcela", "count"), valor_medio=("valor_m2", "mean"), area_mediana=("area_m2", "median"),
              ndvi=("ndvi", "mean"), elevacion=("elevacion_m", "mean"),
              departamento_dominante=("departamento", lambda s: s.mode().iat[0]))
    return t.round(2).reset_index().rename(columns={
        "grupo": "Grupo", "parcelas": "Parcelas", "valor_medio": "Valor medio (US$/m²)",
        "area_mediana": "Área mediana (m²)", "ndvi": "NDVI medio", "elevacion": "Elevación media (m)",
        "departamento_dominante": "Departamento dominante"})


# ---------------------------------------------------------------------------
# 4. ENCABEZADO Y BARRA LATERAL
# ---------------------------------------------------------------------------
st.title("🧭 Agrupamiento Territorial: K-Means y DBSCAN")
st.caption("Demostración de Streamlit + Folium · Curso *Machine Learning Aplicado al Espacio Geográfico* — CNR El Salvador")

with st.sidebar:
    st.header("Configuración")
    preset = st.radio("¿Qué variables usa el modelo?", list(PRESETS), index=1)
    st.caption(PRESETS[preset]["nota"])
    usar_folium = st.toggle("Mapa interactivo (Folium)", value=FOLIUM_OK, disabled=not FOLIUM_OK,
                            help="Si en tu red no se ve el mapa, apágalo para usar un plano de coordenadas.")
    if not FOLIUM_OK:
        st.warning("Falta instalar **folium** y **streamlit-folium** en requirements.txt. Se usa el mapa plano.")
    st.divider()
    deps_sel = st.multiselect("Departamentos a mostrar", list(TERRITORIO), default=list(TERRITORIO))
    st.info("Los modelos se entrenan con **todas** las parcelas; este filtro solo cambia lo que ves en el mapa.")
    st.caption("Datos 100 % sintéticos con fines didácticos.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Parcelas", f"{len(datos):,}")
c2.metric("Departamentos", len(TERRITORIO))
c3.metric("Municipios", sum(len(v) for v in TERRITORIO.values()))
c4.metric("Anomalías inyectadas", int((datos["anomalia"] != "").sum()))

tab_km, tab_db, tab_cmp, tab_datos, tab_como = st.tabs(
    ["🎯 K-Means", "🔍 DBSCAN", "⚖️ Comparación", "📋 Datos", "📘 ¿Cómo funcionan?"])

# ---------------------------------------------------------------------------
# 5. K-MEANS
# ---------------------------------------------------------------------------
with tab_km:
    st.subheader("K-Means: repartir las parcelas en k grupos")
    codo = curva_codo(preset)
    k_sugerido = int(codo.loc[codo["silhouette"].idxmax(), "k"])
    k = st.slider("Número de grupos (k)", 2, 10, PRESETS[preset]["k"],
                  help=f"Punto de partida sugerido para estas variables. El mayor silhouette se da con k = {k_sugerido}.")
    etq, centros_std, inercia, sil, dbi = correr_kmeans(preset, k)
    m1, m2, m3 = st.columns(3)
    m1.metric("Inercia (más baja = grupos más compactos)", f"{inercia:,.0f}")
    m2.metric("Silhouette (más alto = mejor, máx. 1)", f"{sil:.2f}")
    m3.metric("Davies-Bouldin (más bajo = mejor)", f"{dbi:.2f}")

    d_km = datos.assign(cluster=etq)
    # centroides en lat/lon reales (promedio de cada grupo)
    centros = [(d_km.loc[d_km.cluster == i, "latitud"].mean(), d_km.loc[d_km.cluster == i, "longitud"].mean())
               for i in range(k)]
    mostrar_mapa(d_km[d_km["departamento"].isin(deps_sel)], centros)
    st.markdown("**Perfil de cada grupo**")
    st.dataframe(perfil(d_km), hide_index=True, width="stretch")

    st.markdown("**¿Cuántos grupos elegir? Método del codo y silhouette**")
    a, b = st.columns(2)
    f1 = px.line(codo, x="k", y="inercia", markers=True, color_discrete_sequence=[TEAL], height=300,
                 labels={"inercia": "Inercia", "k": "k"}, title="Codo: la curva se aplana al pasar de cierto k")
    f1.add_vline(x=k, line_dash="dash", line_color=GOLD)
    f2 = px.line(codo, x="k", y="silhouette", markers=True, color_discrete_sequence=[NAVY], height=300,
                 labels={"silhouette": "Silhouette", "k": "k"}, title="Silhouette: el pico indica el mejor k")
    f2.add_vline(x=k, line_dash="dash", line_color=GOLD)
    a.plotly_chart(f1, width="stretch"); b.plotly_chart(f2, width="stretch")
    st.info("**Idea clave:** K-Means **siempre** reparte *todas* las parcelas en k grupos, incluso las anómalas. "
            "No tiene noción de «ruido»: por eso lo comparamos con DBSCAN.")

# ---------------------------------------------------------------------------
# 6. DBSCAN
# ---------------------------------------------------------------------------
with tab_db:
    st.subheader("DBSCAN: encontrar zonas densas y separar el ruido")
    P = PRESETS[preset]
    emin, emax, edef, estep = P["eps"]
    col_a, col_b = st.columns(2)
    eps = col_a.slider(f"eps — radio de vecindad ({P['unidad']})", emin, emax, edef, estep)
    min_s = col_b.slider("min_samples — vecinos mínimos para ser zona densa", 2, 20, P["ms"])
    etq_db = correr_dbscan(preset, eps, min_s)
    n_cl, n_ruido, sil_db = metricas_dbscan(preset, etq_db)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grupos encontrados", n_cl)
    m2.metric("Parcelas marcadas como ruido", f"{n_ruido} ({n_ruido / len(datos):.1%})")
    m3.metric("Silhouette (sin ruido)", "—" if np.isnan(sil_db) else f"{sil_db:.2f}")
    inyectadas = datos["anomalia"] != ""
    atrapadas = int(((etq_db == -1) & inyectadas.to_numpy()).sum())
    m4.metric("Anomalías inyectadas detectadas", f"{atrapadas} de {int(inyectadas.sum())}")
    tipos = ["Ubicación aislada", "Valor extremo", "Área extrema"]
    detalle = " · ".join(f"{t}: {int(((etq_db == -1) & (datos['anomalia'] == t).to_numpy()).sum())}/"
                         f"{int((datos['anomalia'] == t).sum())}" for t in tipos)
    st.caption(f"Detección por tipo de anomalía → {detalle}. "
               "Cada receta de variables «ve» un tipo distinto: con *Solo ubicación* solo puede detectar las "
               "parcelas aisladas; con *Atributos del terreno*, los valores y áreas extremos.")

    d_db = datos.assign(cluster=etq_db)
    mostrar_mapa(d_db[d_db["departamento"].isin(deps_sel)])
    if n_ruido:
        st.markdown("**Parcelas marcadas como ruido (posibles anomalías catastrales)**")
        rr = d_db[d_db.cluster == -1][["id_parcela", "departamento", "municipio", "uso_suelo", "area_m2", "valor_m2", "anomalia"]]
        rr = rr.assign(anomalia=rr["anomalia"].replace("", "— (parcela normal)")).round(1)
        st.dataframe(rr, hide_index=True, width="stretch", height=240)

    st.markdown("**¿Cómo elegir eps? Gráfico de k-distancia**")
    kd = curva_k_distancia(preset, min_s)
    fkd = px.line(y=kd, labels={"x": "Parcelas ordenadas por distancia", "y": f"Distancia al vecino n.º {min_s}"},
                  color_discrete_sequence=[TEAL], height=320)
    fkd.add_hline(y=eps, line_dash="dash", line_color=GOLD, annotation_text="eps elegido")
    st.plotly_chart(fkd, width="stretch")
    st.caption("Se busca el «codo» donde la curva se dispara: a partir de ahí están las parcelas aisladas.")
    st.info("**Prueba en vivo:** sube *eps* y los grupos se funden en uno; bájalo y todo se vuelve ruido. "
            "Sube *min_samples* y las zonas pequeñas desaparecen.")

# ---------------------------------------------------------------------------
# 7. COMPARACIÓN
# ---------------------------------------------------------------------------
with tab_cmp:
    st.subheader("K-Means vs. DBSCAN con las mismas variables")
    _, _, in_km, sil_km, dbi_km = correr_kmeans(preset, k)
    tabla = pd.DataFrame([
        ["K-Means", f"k = {k}", k, 0, f"{sil_km:.2f}", "No (todo parcela va a un grupo)", "Redondos, tamaño parecido"],
        ["DBSCAN", f"eps = {eps}, min_samples = {min_s}", n_cl, n_ruido,
         "—" if np.isnan(sil_db) else f"{sil_db:.2f}", "Sí (etiqueta −1)", "Cualquier forma"],
    ], columns=["Modelo", "Parámetros", "Grupos", "Parcelas ruido", "Silhouette", "¿Detecta anomalías?", "Forma de los grupos"])
    st.dataframe(tabla, hide_index=True, width="stretch")
    cuenta = pd.DataFrame({"Grupo": [nombre_cluster(c) for c in sorted(set(etq_db))],
                           "Parcelas": [int((etq_db == c).sum()) for c in sorted(set(etq_db))]})
    fb = px.bar(cuenta, x="Grupo", y="Parcelas", color="Grupo", height=320,
                color_discrete_map={nombre_cluster(c): color_cluster(c) for c in set(etq_db)}, title="Tamaño de los grupos de DBSCAN")
    fb.update_layout(showlegend=False)
    st.plotly_chart(fb, width="stretch")
    st.markdown("""
| Pregunta | K-Means | DBSCAN |
|---|---|---|
| ¿Debo decir cuántos grupos quiero? | **Sí** (k) | **No**, los descubre |
| ¿Qué parámetros pide? | k | eps y min_samples |
| ¿Maneja anomalías? | No | **Sí**, las marca como ruido |
| ¿Sirve para zonas alargadas o irregulares? | Poco | **Muy bien** |
| Caso CNR típico | Zonificación en k zonas de valor | Detectar parcelas atípicas y zonas densas |
""")
    st.success("Prueba cambiar de variables en la barra lateral: con **Solo ubicación** DBSCAN aísla parcelas lejanas; "
               "con **Atributos del terreno** detecta valores y áreas extremas.")

# ---------------------------------------------------------------------------
# 8. DATOS
# ---------------------------------------------------------------------------
with tab_datos:
    st.subheader("Datos y resultados")
    res = datos.assign(grupo_kmeans=[nombre_cluster(c) for c in etq],
                       grupo_dbscan=[nombre_cluster(c) for c in etq_db])
    res = res[res["departamento"].isin(deps_sel)].drop(columns=["x_km", "y_km"]).round(3)
    st.dataframe(res, hide_index=True, width="stretch", height=420)
    st.download_button("⬇️ Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                       "parcelas_agrupadas.csv", "text/csv")
    cnt = datos.groupby(["departamento", "municipio"]).size().reset_index(name="parcelas")
    with st.expander("Parcelas por departamento y municipio"):
        st.dataframe(cnt, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 9. CÓMO FUNCIONAN
# ---------------------------------------------------------------------------
with tab_como:
    st.subheader("Los dos modelos en lenguaje sencillo")
    a, b = st.columns(2)
    with a:
        st.markdown("### 🎯 K-Means")
        st.markdown("**Analogía:** repartir *k* oficinas regionales del CNR. Colocas las oficinas al azar, cada "
                    "parcela se asigna a la oficina más cercana, luego cada oficina se mueve al centro de sus "
                    "parcelas, y se repite hasta que nadie cambia de oficina.")
        st.code("""from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(df[["x_km", "y_km", "valor_m2"]])
modelo = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X)
df["grupo"] = modelo.labels_""", language="python")
    with b:
        st.markdown("### 🔍 DBSCAN")
        st.markdown("**Analogía:** un incendio que se propaga. Si una parcela tiene al menos *min_samples* vecinos "
                    "dentro del radio *eps*, es un foco; el fuego salta a sus vecinos y forma una zona. "
                    "Las parcelas a las que el fuego nunca llega son **ruido**.")
        st.code("""from sklearn.cluster import DBSCAN

modelo = DBSCAN(eps=2.5, min_samples=5).fit(df[["x_km", "y_km"]])
df["grupo"] = modelo.labels_      # -1 = ruido / anomalía""", language="python")
    st.markdown("### Pasos que sigue esta app")
    st.markdown("""
1. **Datos:** 1,500 parcelas sintéticas, 8 departamentos × 4 municipios, con 45 anomalías inyectadas.
2. **Preparación:** se eligen las variables y se **estandarizan** (para que el valor en dólares no pese más que el NDVI).
3. **Entrenamiento:** K-Means y DBSCAN se entrenan al mover los controles (resultados guardados con `@st.cache_data`).
4. **Evaluación:** inercia, silhouette y Davies-Bouldin; en DBSCAN, además, cuántas anomalías reales detectó.
5. **Decisión:** el mapa muestra los grupos para zonificar y el ruido para revisión de campo.
""")
    st.caption("Facilitadora: Jessica Martínez · doulus.jefis@gmail.com · Datos sintéticos, solo con fines educativos.")
