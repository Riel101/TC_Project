import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Outbreak Clustering",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1224 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8 !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f1e38 0%, #0a1628 50%, #091020 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}

/* ── Metric cards ── */
.metric-card {
    background: #0d1224;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    display: block;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* ── Section headings ── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 600;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 1.2rem;
}

/* ── Cluster badges ── */
.cluster-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem;
}
.badge-0  { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid #38bdf8; }
.badge-1  { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid #818cf8; }
.badge-2  { background: rgba(249,115,22,0.15);  color: #fb923c; border: 1px solid #fb923c; }
.badge-noise { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid #64748b; }

/* ── Tabs ── */
[data-testid="stTab"] button {
    color: #64748b !important;
    font-weight: 500;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8;
}

/* ── Data tables ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Alerts/info boxes ── */
.info-box {
    background: rgba(56,189,248,0.06);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.9rem;
}
.warn-box {
    background: rgba(251,191,36,0.06);
    border-left: 3px solid #fbbf24;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.9rem;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #1e3a5f !important;
    border-radius: 12px !important;
    background: #0d1224 !important;
    padding: 1rem !important;
}

/* ── Divider ── */
hr { border-color: #1e2d4a !important; }

/* ── Number inputs, selects ── */
[data-testid="stNumberInput"], [data-testid="stSelectbox"] {
    background: #0d1224;
}

/* ── Spinner ── */
.stSpinner { color: #38bdf8 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #0f1e38);
    color: #38bdf8;
    border: 1px solid #38bdf8;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    color: #0a0e1a;
    border-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK STYLE
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1224",
    "axes.facecolor":    "#0d1224",
    "axes.edgecolor":    "#1e2d4a",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e2d4a",
    "grid.linestyle":    "--",
    "axes.grid":         True,
    "legend.facecolor":  "#0a0e1a",
    "legend.edgecolor":  "#1e2d4a",
    "figure.dpi":        120,
})

PALETTE   = ["#38bdf8", "#818cf8", "#fb923c", "#34d399", "#f472b6"]
NOISE_COL = "#475569"


# ─────────────────────────────────────────────
#  DATA PIPELINE  (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(uploaded_bytes))
    cols_to_drop = [
        "iso2", "icd10c", "icd103c", "icd104c",
        "icd11c1", "icd11c2", "icd11c3",
        "icd104n", "icd11l1", "icd11l2", "icd11l3",
        "DONs", "Definition",
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    df_clean = df.drop(columns=existing)
    return df_clean


@st.cache_data(show_spinner=False)
def engineer_features(df_clean: pd.DataFrame):
    country_features = df_clean.groupby("Country").agg(
        total_outbreaks    = ("Disease", "count"),
        unique_diseases    = ("Disease", "nunique"),
        year_span          = ("Year", lambda x: x.max() - x.min()),
        avg_year           = ("Year", "mean"),
        most_common_category = ("icd10n", lambda x: x.mode()[0]),
    ).reset_index()

    le = LabelEncoder()
    country_features["category_encoded"] = le.fit_transform(
        country_features["most_common_category"]
    )
    encoding_map = {i: lbl for i, lbl in enumerate(le.classes_)}
    return country_features, encoding_map


@st.cache_data(show_spinner=False)
def scale_features(country_features: pd.DataFrame):
    features = ["total_outbreaks", "unique_diseases", "year_span", "avg_year", "category_encoded"]
    X = country_features[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, features


@st.cache_data(show_spinner=False)
def run_kmeans(X_scaled, k: int):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels)
    return km, labels, sil


@st.cache_data(show_spinner=False)
def run_dbscan(X_scaled, eps: float, min_samples: int):
    db     = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    sil = None
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            sil = silhouette_score(X_scaled[mask], labels[mask])
    return labels, n_clusters, n_noise, sil


@st.cache_data(show_spinner=False)
def compute_pca(X_scaled):
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_


@st.cache_data(show_spinner=False)
def elbow_data(X_scaled):
    inertias = []
    k_range  = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return list(k_range), inertias


# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────
def plot_elbow(k_range, inertias):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_range, inertias, marker="o", color="#38bdf8", linewidth=2.5, markersize=7,
            markerfacecolor="#0d1224", markeredgewidth=2)
    ax.fill_between(k_range, inertias, alpha=0.08, color="#38bdf8")
    ax.set_title("Elbow Method — Inertia vs K", fontsize=13, fontweight="bold", color="#e2e8f0", pad=12)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(k_range)
    fig.tight_layout()
    return fig


def plot_pca_scatter(X_pca, labels, title, ev):
    unique_labels = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = NOISE_COL if lbl == -1 else PALETTE[i % len(PALETTE)]
        name  = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, s=70, alpha=0.85,
            edgecolors="#0d1224", linewidths=0.5,
            label=name,
        )
    ax.set_title(title, fontsize=13, fontweight="bold", color="#e2e8f0", pad=12)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.legend(framealpha=0.5)
    fig.tight_layout()
    return fig


def plot_cluster_bar(labels, title):
    from collections import Counter
    counts = Counter(labels)
    keys   = sorted(counts.keys())
    vals   = [counts[k] for k in keys]
    colors = [NOISE_COL if k == -1 else PALETTE[k % len(PALETTE)] for k in keys]
    x_labels = ["Noise" if k == -1 else f"Cluster {k}" for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x_labels, vals, color=colors, edgecolor="#0d1224", linewidth=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=11, color="#e2e8f0", fontweight="600")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#e2e8f0", pad=10)
    ax.set_ylabel("Number of Countries")
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    return fig


def plot_yearly_bar(df_clean):
    yc = df_clean["Year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(yc.index, yc.values, color="#38bdf8", edgecolor="#0d1224", linewidth=0.5, width=0.8)
    ax.set_title("Disease Outbreaks per Year", fontsize=12, fontweight="bold", color="#e2e8f0", pad=10)
    ax.set_xlabel("Year"); ax.set_ylabel("Outbreaks")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_top_countries(df_clean, n=20):
    tc = df_clean["Country"].value_counts().head(n)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(tc))]
    ax.barh(tc.index[::-1], tc.values[::-1], color=colors[::-1], edgecolor="#0d1224", linewidth=0.5)
    ax.set_title(f"Top {n} Countries by Outbreak Count", fontsize=12, fontweight="bold", color="#e2e8f0", pad=10)
    ax.set_xlabel("Outbreaks")
    fig.tight_layout()
    return fig


def plot_top_diseases(df_clean, n=15):
    td = df_clean["Disease"].value_counts().head(n)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(td.index[::-1], td.values[::-1], color="#818cf8", edgecolor="#0d1224", linewidth=0.5)
    ax.set_title(f"Top {n} Most Frequent Diseases", fontsize=12, fontweight="bold", color="#e2e8f0", pad=10)
    ax.set_xlabel("Count")
    fig.tight_layout()
    return fig


def plot_icd_pie(df_clean):
    cat_counts = df_clean["icd10n"].value_counts()
    top_n, top_cats = 6, cat_counts.head(6)
    other = pd.Series({"Other": cat_counts.iloc[top_n:].sum()})
    plot_data = pd.concat([top_cats, other])
    fig, ax = plt.subplots(figsize=(9, 6))
    wedge_colors = PALETTE + ["#94a3b8"]
    wedges, texts, autotexts = ax.pie(
        plot_data.values, labels=plot_data.index,
        autopct="%1.1f%%", startangle=140,
        colors=wedge_colors[:len(plot_data)],
        pctdistance=0.78, labeldistance=1.1,
        wedgeprops={"edgecolor": "#0a0e1a", "linewidth": 1.5},
    )
    for t in texts: t.set_color("#cbd5e1"); t.set_fontsize(9)
    for at: at.set_color("#0a0e1a"); at.set_fontweight("bold"); at.set_fontsize(8)
    ax.set_title("ICD-10 Disease Category Distribution", fontsize=12, fontweight="bold",
                 color="#e2e8f0", pad=14)
    fig.tight_layout()
    return fig


def cluster_profile_table(country_features, labels, features):
    cf = country_features.copy()
    cf["cluster"] = labels
    profile = cf.groupby("cluster")[features].mean().round(2)
    profile.index = [f"Cluster {i}" if i != -1 else "Noise" for i in profile.index]
    return profile


CLUSTER_DESCRIPTIONS = {
    0: ("Low Burden", "#38bdf8",
        "Countries with few outbreaks and limited disease diversity. "
        "Likely isolated or well-resourced nations."),
    1: ("Moderate Burden", "#818cf8",
        "Countries with intermediate outbreak frequency spanning longer timelines. "
        "Diverse regional profiles."),
    2: ("High Burden", "#fb923c",
        "High-frequency, highly diverse outbreak environments. Typically Sub-Saharan "
        "African nations facing chronic multi-disease challenges."),
}


# ─────────────────────────────────────────────
#  PREDICT HELPER
# ─────────────────────────────────────────────
def predict_cluster(kmeans_model, scaler, total_outbreaks, unique_diseases,
                    year_span, avg_year, category_encoded):
    x = np.array([[total_outbreaks, unique_diseases, year_span, avg_year, category_encoded]])
    x_scaled = scaler.transform(x)
    label = kmeans_model.predict(x_scaled)[0]
    return label


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Disease Outbreak\nClustering Dashboard")
    st.markdown("---")

    st.markdown("### 📂 Data")
    uploaded = st.file_uploader("Upload `outbreaks.csv`", type=["csv"])

    st.markdown("---")
    st.markdown("### ⚙️ K-Means Parameters")
    optimal_k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)

    st.markdown("### ⚙️ DBSCAN Parameters")
    eps_val        = st.slider("eps (neighbourhood radius)", 0.3, 3.0, 1.2, 0.1)
    min_samples_val= st.slider("min_samples", 2, 10, 3)

    st.markdown("---")
    st.markdown("### 📑 Navigation")
    page = st.radio(
        "",
        ["Overview", "EDA", "Modelling", "Evaluation", "Predict Country"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Group 38 · Disease Outbreak Clustering · WHO Dataset 1996–2026")


# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🧬 Disease Outbreak Clustering</p>
  <p class="hero-sub">
    WHO Global Outbreak Dataset (1996–2026) &nbsp;·&nbsp;
    K-Means &amp; DBSCAN Clustering &nbsp;·&nbsp;
    Country-Level Risk Stratification
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  GATE — require CSV upload
# ─────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="info-box">
        👆 Upload your <code>outbreaks.csv</code> file in the sidebar to get started.
        The app will automatically clean, engineer features, and run both clustering models.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────
with st.spinner("Loading & cleaning data…"):
    df_raw   = load_and_clean(uploaded.read())

with st.spinner("Engineering features…"):
    country_features, encoding_map = engineer_features(df_raw)

with st.spinner("Scaling features…"):
    X_scaled, scaler, features = scale_features(country_features)

with st.spinner("Training K-Means…"):
    kmeans_model, kmeans_labels, sil_kmeans = run_kmeans(X_scaled, optimal_k)
    country_features["kmeans_cluster"] = kmeans_labels

with st.spinner("Training DBSCAN…"):
    dbscan_labels, n_db_clusters, n_noise, sil_dbscan = run_dbscan(X_scaled, eps_val, min_samples_val)
    country_features["dbscan_cluster"] = dbscan_labels

with st.spinner("Computing PCA…"):
    X_pca, ev = compute_pca(X_scaled)

k_range, inertias = elbow_data(X_scaled)


# ─────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (len(df_raw), "Total Records"),
        (df_raw["Country"].nunique(), "Countries"),
        (df_raw["Disease"].nunique(), "Unique Diseases"),
        (int(df_raw["Year"].max() - df_raw["Year"].min()), "Year Span"),
        (f"{sil_kmeans:.4f}", "K-Means Silhouette"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <span class="metric-val">{val}</span>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cluster cards
    st.markdown('<div class="section-heading">K-Means Cluster Summary</div>', unsafe_allow_html=True)
    cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()

    cols = st.columns(optimal_k)
    for i, col in enumerate(cols):
        n   = cluster_counts.get(i, 0)
        desc = CLUSTER_DESCRIPTIONS.get(i, (f"Cluster {i}", "#38bdf8", ""))
        label, color, text = desc
        countries_in = country_features[country_features["kmeans_cluster"] == i]["Country"].head(5).tolist()
        sample_str = ", ".join(countries_in)
        col.markdown(f"""
        <div class="metric-card" style="text-align:left; padding:1.4rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.7rem;">
                <div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>
                <span style="color:{color};font-weight:700;font-size:0.9rem;">Cluster {i} — {label}</span>
            </div>
            <div style="font-size:1.8rem;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;">{n}</div>
            <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">countries</div>
            <div style="color:#94a3b8;font-size:0.82rem;line-height:1.5;">{text}</div>
            <div style="margin-top:0.8rem;color:#64748b;font-size:0.78rem;">
                <strong style="color:#475569;">Sample:</strong> {sample_str}…
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick model comparison
    st.markdown('<div class="section-heading">Model Comparison at a Glance</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        sil_db_str = f"{sil_dbscan:.4f}" if sil_dbscan else "N/A"
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div style="color:#38bdf8;font-weight:700;font-size:0.9rem;margin-bottom:0.8rem;">🔵 K-Means (K={optimal_k})</div>
            <div>Inertia: <strong style="color:#38bdf8;font-family:'JetBrains Mono',monospace;">{kmeans_model.inertia_:.2f}</strong></div>
            <div>Silhouette: <strong style="color:#38bdf8;font-family:'JetBrains Mono',monospace;">{sil_kmeans:.4f}</strong></div>
            <div>Noise Points: <strong style="color:#38bdf8;font-family:'JetBrains Mono',monospace;">0</strong></div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div style="color:#818cf8;font-weight:700;font-size:0.9rem;margin-bottom:0.8rem;">🟣 DBSCAN (eps={eps_val}, min_samples={min_samples_val})</div>
            <div>Clusters Found: <strong style="color:#818cf8;font-family:'JetBrains Mono',monospace;">{n_db_clusters}</strong></div>
            <div>Silhouette: <strong style="color:#818cf8;font-family:'JetBrains Mono',monospace;">{sil_db_str}</strong></div>
            <div>Noise Points: <strong style="color:#818cf8;font-family:'JetBrains Mono',monospace;">{n_noise}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Full Country Cluster Table</div>', unsafe_allow_html=True)
    display_df = country_features[["Country", "total_outbreaks", "unique_diseases",
                                   "year_span", "avg_year", "most_common_category",
                                   "kmeans_cluster", "dbscan_cluster"]].copy()
    display_df.columns = ["Country", "Total Outbreaks", "Unique Diseases",
                          "Year Span", "Avg Year", "Top Disease Category",
                          "K-Means Cluster", "DBSCAN Cluster"]
    st.dataframe(display_df.sort_values("Total Outbreaks", ascending=False), use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE: EDA
# ─────────────────────────────────────────────
elif page == "EDA":
    st.markdown('<div class="section-heading">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📅 Outbreaks Over Time", "🌍 Countries", "🦠 Diseases", "📊 ICD-10 Categories"])

    with tab1:
        st.markdown("""
        <div class="info-box">
        Notable spike around 2020–2021 corresponding to the COVID-19 pandemic.
        This confirms <code>Year</code> as a meaningful clustering feature.
        </div>""", unsafe_allow_html=True)
        st.pyplot(plot_yearly_bar(df_raw), use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="info-box">
        Sub-Saharan African nations dominate the outbreak counts, alongside large nations
        like the USA and China — informing distinct regional cluster groupings.
        </div>""", unsafe_allow_html=True)
        n_countries = st.slider("Show top N countries", 5, 40, 20)
        st.pyplot(plot_top_countries(df_raw, n_countries), use_container_width=True)

    with tab3:
        st.markdown("""
        <div class="info-box">
        COVID-19 and pandemic influenza dominate. Cholera, polio, and yellow fever are
        consistently high. This imbalance guided the choice of Label Encoding over one-hot.
        </div>""", unsafe_allow_html=True)
        n_diseases = st.slider("Show top N diseases", 5, 30, 15)
        st.pyplot(plot_top_diseases(df_raw, n_diseases), use_container_width=True)

    with tab4:
        st.markdown("""
        <div class="info-box">
        Most outbreaks fall under a few broad ICD-10 categories, which reduces noise
        compared to using individual disease names as a feature.
        </div>""", unsafe_allow_html=True)
        if "icd10n" in df_raw.columns:
            st.pyplot(plot_icd_pie(df_raw), use_container_width=True)
        else:
            st.warning("Column `icd10n` not found in the dataset.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Encoding Map — ICD-10 Category → Integer</div>',
                unsafe_allow_html=True)
    enc_df = pd.DataFrame(
        [(k, v) for k, v in encoding_map.items()],
        columns=["Encoded Value", "Disease Category"]
    )
    st.dataframe(enc_df, use_container_width=True, height=250)


# ─────────────────────────────────────────────
#  PAGE: MODELLING
# ─────────────────────────────────────────────
elif page == "Modelling":
    st.markdown('<div class="section-heading">Model Training & Diagnostics</div>',
                unsafe_allow_html=True)

    tab_km, tab_db = st.tabs(["🔵 K-Means", "🟣 DBSCAN"])

    # ── K-Means ──
    with tab_km:
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.markdown("**Elbow Method — choosing optimal K**")
            st.pyplot(plot_elbow(k_range, inertias), use_container_width=True)
        with col2:
            st.markdown("""
            <div class="info-box">
            The <strong>Elbow Method</strong> plots inertia against K.
            The "elbow" point — where the curve bends — marks where additional
            clusters yield diminishing returns.
            Use the sidebar slider to experiment with different K values.
            </div>""", unsafe_allow_html=True)

            st.markdown("**Selected K metrics**")
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <div>K = <strong style="color:#38bdf8;">{optimal_k}</strong></div>
                <div>Inertia = <strong style="color:#38bdf8;font-family:'JetBrains Mono',monospace;">{kmeans_model.inertia_:.2f}</strong></div>
                <div>Silhouette = <strong style="color:#38bdf8;font-family:'JetBrains Mono',monospace;">{sil_kmeans:.4f}</strong></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("**PCA 2D Projection — K-Means Clusters**")
        st.pyplot(
            plot_pca_scatter(X_pca, kmeans_labels,
                             f"K-Means Clustering (K={optimal_k}) — PCA 2D", ev),
            use_container_width=True,
        )

    # ── DBSCAN ──
    with tab_db:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <div style="color:#818cf8;font-weight:700;margin-bottom:0.5rem;">DBSCAN Results</div>
                <div>eps = <strong style="color:#818cf8;">{eps_val}</strong></div>
                <div>min_samples = <strong style="color:#818cf8;">{min_samples_val}</strong></div>
                <div>Clusters Found = <strong style="color:#818cf8;font-family:'JetBrains Mono',monospace;">{n_db_clusters}</strong></div>
                <div>Noise Points = <strong style="color:#818cf8;font-family:'JetBrains Mono',monospace;">{n_noise}</strong></div>
                <div>Silhouette = <strong style="color:#818cf8;font-family:'JetBrains Mono',monospace;">{sil_dbscan:.4f if sil_dbscan else "N/A"}</strong></div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="warn-box">
            <strong>DBSCAN Sensitivity:</strong> Adjust <code>eps</code> and
            <code>min_samples</code> in the sidebar to tune clustering.
            Very low eps → many noise points. Very high eps → one large cluster.
            </div>""", unsafe_allow_html=True)

        st.markdown("**PCA 2D Projection — DBSCAN Clusters**")
        st.pyplot(
            plot_pca_scatter(X_pca, dbscan_labels,
                             "DBSCAN Clustering — PCA 2D", ev),
            use_container_width=True,
        )

        # Noise country list
        if n_noise > 0:
            noise_countries = country_features[country_features["dbscan_cluster"] == -1]["Country"].tolist()
            st.markdown(f"**Noise / Outlier Countries ({n_noise})**")
            st.markdown(" ".join(
                f'<span class="cluster-badge badge-noise">{c}</span>'
                for c in noise_countries
            ), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: EVALUATION
# ─────────────────────────────────────────────
elif page == "Evaluation":
    st.markdown('<div class="section-heading">Model Evaluation</div>', unsafe_allow_html=True)

    # Side-by-side cluster distributions
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(
            plot_cluster_bar(kmeans_labels, f"K-Means — Countries per Cluster (K={optimal_k})"),
            use_container_width=True,
        )
    with c2:
        st.pyplot(
            plot_cluster_bar(dbscan_labels, "DBSCAN — Cluster Distribution"),
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Cluster Profile — Average Feature Values</div>',
                unsafe_allow_html=True)

    tab_kp, tab_dp = st.tabs(["K-Means Profiles", "DBSCAN Profiles"])
    with tab_kp:
        kp = cluster_profile_table(country_features, kmeans_labels, features)
        st.dataframe(kp.style.format("{:.2f}"), use_container_width=True)
    with tab_dp:
        dp = cluster_profile_table(country_features, dbscan_labels, features)
        st.dataframe(dp.style.format("{:.2f}"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Final Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    ✅ <strong>K-Means (K={optimal_k})</strong> is recommended for this dataset.
    It produces a more actionable stratification of countries by public health risk,
    backed by a superior silhouette score
    (<strong style="color:#38bdf8;">{sil_kmeans:.4f}</strong> vs
    <strong style="color:#818cf8;">{f"{sil_dbscan:.4f}" if sil_dbscan else "N/A"}</strong> for DBSCAN)
    and cleaner PCA separation.
    DBSCAN is useful for outlier detection ({n_noise} noise countries identified).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Limitations</div>', unsafe_allow_html=True)
    limitations = [
        "Dataset records occurrence only — not case counts or deaths, limiting severity analysis.",
        "Country-level granularity only; sub-national outbreak patterns are invisible.",
        "Label encoding introduces implicit ordinality that may not reflect true disease similarity.",
        "ICD-10 label frequencies are imbalanced, potentially biasing the `most_common_category` feature.",
    ]
    for lim in limitations:
        st.markdown(f"<div class='warn-box'>⚠️ {lim}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: PREDICT COUNTRY
# ─────────────────────────────────────────────
elif page == "Predict Country":
    st.markdown('<div class="section-heading">🔍 Predict Cluster for a Country Profile</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Enter the outbreak profile for a country (real or hypothetical) to predict
    which K-Means cluster it would fall into.
    </div>""", unsafe_allow_html=True)

    # Option A: pick existing country
    country_list = sorted(country_features["Country"].tolist())
    selected_country = st.selectbox("Load from existing country (optional)", ["— Enter manually —"] + country_list)

    if selected_country != "— Enter manually —":
        row = country_features[country_features["Country"] == selected_country].iloc[0]
        default_to  = int(row["total_outbreaks"])
        default_ud  = int(row["unique_diseases"])
        default_ys  = int(row["year_span"])
        default_ay  = float(row["avg_year"])
        default_ce  = int(row["category_encoded"])
        actual_cluster = int(row["kmeans_cluster"])
    else:
        default_to, default_ud, default_ys, default_ay, default_ce, actual_cluster = 10, 5, 15, 2010.0, 0, None

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        total_outbreaks  = st.number_input("Total Outbreaks", min_value=1, value=default_to)
        unique_diseases  = st.number_input("Unique Diseases", min_value=1, value=default_ud)
    with col2:
        year_span        = st.number_input("Year Span (years)", min_value=0, value=default_ys)
        avg_year         = st.number_input("Average Year", min_value=1990.0, max_value=2030.0,
                                           value=default_ay, step=0.5)
    with col3:
        st.markdown("**ICD-10 Category Encoding**")
        cat_options = [(v, k) for k, v in encoding_map.items()]
        cat_display = [f"{label} (→{code})" for label, code in cat_options]
        cat_idx     = st.selectbox("Most Common Category", range(len(cat_display)),
                                   format_func=lambda i: cat_display[i],
                                   index=min(default_ce, len(cat_display)-1))
        category_encoded = cat_options[cat_idx][1]

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Cluster", use_container_width=False):
        pred = predict_cluster(kmeans_model, scaler,
                               total_outbreaks, unique_diseases,
                               year_span, avg_year, category_encoded)
        desc = CLUSTER_DESCRIPTIONS.get(pred, (f"Cluster {pred}", "#38bdf8", ""))
        label, color, text = desc

        st.markdown(f"""
        <div class="metric-card" style="max-width:500px;text-align:left;padding:1.8rem;">
            <div style="font-size:0.8rem;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.6rem;">Predicted Cluster</div>
            <div style="font-size:2.5rem;font-weight:700;color:{color};
                        font-family:'JetBrains Mono',monospace;">Cluster {pred}</div>
            <div style="color:{color};font-weight:600;margin:0.4rem 0 0.8rem;">{label}</div>
            <div style="color:#94a3b8;font-size:0.9rem;line-height:1.6;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

        if actual_cluster is not None:
            match = actual_cluster == pred
            icon  = "✅" if match else "⚠️"
            msg   = (f"{icon} Matches the actual cluster ({actual_cluster}) from the training data."
                     if match else
                     f"{icon} Differs from actual training cluster ({actual_cluster}). "
                     "This may be due to rounding or parameter changes.")
            st.markdown(f'<div class="{"info-box" if match else "warn-box"}">{msg}</div>',
                        unsafe_allow_html=True)

        # Show all cluster distances (soft distances via scaled input)
        x = np.array([[total_outbreaks, unique_diseases, year_span, avg_year, category_encoded]])
        x_scaled = scaler.transform(x)
        distances = np.linalg.norm(kmeans_model.cluster_centers_ - x_scaled, axis=1)
        dist_df = pd.DataFrame({
            "Cluster": [f"Cluster {i}" for i in range(optimal_k)],
            "Distance to Centroid": distances.round(4),
        }).sort_values("Distance to Centroid")
        st.markdown("**Distance to each centroid:**")
        st.dataframe(dist_df, use_container_width=True, hide_index=True)