import os
os.environ["OMP_NUM_THREADS"] = "1"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from clustering_helper import (
    load_dataset,
    get_numeric_columns,
    get_safe_default_features,
    prepare_feature_matrix,
    calculate_elbow,
    fit_kmeans_model,
    generate_cluster_insights,
)

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="SegmentIQ | Customer Segmentation Studio",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Modern CSS Design System: Ultra-Premium Glassmorphic Theme
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #F8FAFC;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* Streamlit Top Header Bar */
    header[data-testid="stHeader"] {
        background: rgba(9, 13, 22, 0.75) !important;
        backdrop-filter: blur(10px) !important;
        height: 3.2rem !important;
    }

    /* Main Container Padding - Ensures header elements are never clipped */
    .block-container {
        padding-top: 5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1400px;
    }

    /* ---------------- Sidebar Styling ---------------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B0F19 0%, #0F172A 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 3.5rem;
    }

    /* Brand Header Box in Sidebar */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 16px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .brand-icon-box {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    }
    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .brand-badge {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        background: rgba(99, 102, 241, 0.3);
        color: #A5B4FC;
        padding: 2px 7px;
        border-radius: 6px;
        margin-left: 6px;
        border: 1px solid rgba(99, 102, 241, 0.5);
    }
    .brand-sub {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Sidebar Section Card */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }
    .sidebar-card-header {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }

    /* Multiselect Tags Overrides */
    span[data-baseweb="tag"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 8px !important;
        color: #E0E7FF !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 4px 8px !important;
    }
    span[data-baseweb="tag"] span {
        color: #E0E7FF !important;
    }

    /* ---------------- Main Hero Header ---------------- */
    .hero-container {
        padding-top: 0.6rem;
        margin-bottom: 0.5rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.22) 0%, rgba(6, 182, 212, 0.16) 100%);
        border: 1px solid rgba(99, 102, 241, 0.45);
        color: #C7D2FE;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 30%, #C7D2FE 70%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.5;
        max-width: 950px;
        margin-bottom: 18px;
    }

    /* Meta Chips Bar */
    .meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 24px;
        padding: 10px 16px;
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        align-items: center;
    }
    .meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        color: #CBD5E1;
        background: rgba(255, 255, 255, 0.04);
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .meta-chip strong {
        color: #F8FAFC;
    }

    /* ---------------- KPI Metric Cards ---------------- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
        margin-bottom: 26px;
    }
    .kpi-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #06B6D4);
        opacity: 0.85;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.35);
    }
    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
    }
    .kpi-icon-badge {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: rgba(99, 102, 241, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .kpi-pill-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .kpi-pill-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .kpi-pill-indigo {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* ---------------- Tab Customization ---------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(17, 24, 39, 0.6);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.9rem;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(99, 102, 241, 0.45) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* ---------------- Persona Cards ---------------- */
    .persona-card-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 16px;
        margin-top: 14px;
    }
    .persona-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 5px solid #6366F1;
        border-radius: 14px;
        padding: 20px 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .persona-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .persona-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .persona-strategy-box {
        margin-top: 12px;
        padding: 12px 16px;
        border-radius: 10px;
        background: rgba(17, 24, 39, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.9rem;
    }

    /* ---------------- Callout Info Box ---------------- */
    .info-callout {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(6, 182, 212, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 16px;
        font-size: 0.9rem;
        color: #CBD5E1;
    }

    /* ---------------- Download Button ---------------- */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    div.stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Matplotlib Figure Dark Theme Helper
# ---------------------------------------------------------
def apply_dark_theme(fig, ax_list):
    """
    Applies an ultra-sleek obsidian/slate dark theme to Matplotlib axes.
    Ensures seamless blending with the Streamlit dark dashboard.
    """
    fig.patch.set_facecolor('#0B0F19')
    
    if not isinstance(ax_list, (list, np.ndarray)):
        ax_list = [ax_list]
        
    for ax in ax_list:
        ax.set_facecolor('#111827')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(colors='#94A3B8', labelsize=9.5)
        ax.xaxis.label.set_color('#E2E8F0')
        ax.yaxis.label.set_color('#E2E8F0')
        ax.title.set_color('#FFFFFF')
        ax.grid(True, linestyle='--', color='#1E293B', alpha=0.7)


# ---------------------------------------------------------
# Sidebar: Control Hub
# ---------------------------------------------------------
with st.sidebar:
    # Sleek Brand Banner
    st.markdown("""
    <div class="brand-container">
        <div class="brand-icon-box">🛍️</div>
        <div>
            <div class="brand-title">SegmentIQ <span class="brand-badge">STUDIO</span></div>
            <div class="brand-sub">K-Means Customer Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Data Source
    st.markdown("""
    <div class="sidebar-card-header">
        <span>📁 Data Stream</span>
        <span style="color: #34D399; font-size: 0.7rem;">● Ready</span>
    </div>
    """, unsafe_allow_html=True)

    data_source_mode = st.radio(
        "Select Dataset Source:",
        ["Bundled Mall Dataset (Mall_Customers.csv)", "Upload Custom CSV File"],
        index=0,
        label_visibility="collapsed"
    )

    df = None
    data_err = None
    active_data_label = ""

    if data_source_mode == "Bundled Mall Dataset (Mall_Customers.csv)":
        default_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
        if os.path.exists(default_path):
            df, data_err = load_dataset(default_path)
            active_data_label = "Mall_Customers.csv (Default)"
        else:
            data_err = "Default dataset 'Mall_Customers.csv' was not found in the project root."
    else:
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            df, data_err = load_dataset(uploaded_file)
            active_data_label = f"Uploaded: {uploaded_file.name}"
        else:
            st.info("👋 Please upload a CSV file or switch to the bundled dataset above.")
            st.stop()

    if data_err:
        st.error(f"❌ {data_err}")
        st.stop()

    if df is None or df.empty:
        st.warning("⚠️ No data available to display.")
        st.stop()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Section 2: Features for Clustering
    numeric_columns = get_numeric_columns(df, exclude_id=True)
    if len(numeric_columns) < 2:
        st.error(f"❌ Found only {len(numeric_columns)} numeric column(s). At least 2 are required.")
        st.stop()

    safe_defaults = get_safe_default_features(numeric_columns)

    st.markdown("""
    <div class="sidebar-card-header">
        <span>⚙️ Feature Space</span>
        <span style="color: #818CF8; font-size: 0.7rem;">2D Projection</span>
    </div>
    """, unsafe_allow_html=True)

    feature_widget_key = f"feat_sel_{active_data_label}_{hash(tuple(numeric_columns))}"
    selected_features = st.multiselect(
        "Choose exactly 2 numeric features:",
        options=numeric_columns,
        default=safe_defaults,
        key=feature_widget_key,
        help="Select two continuous numeric features for clustering and 2D visualization."
    )

    if len(selected_features) != 2:
        st.warning(f"⚠️ Please select exactly 2 numerical features (selected: {len(selected_features)}).")
        st.stop()

    # Prepare feature matrix
    X_clean, valid_mask, prep_err = prepare_feature_matrix(df, selected_features)
    if prep_err:
        st.error(f"❌ {prep_err}")
        st.stop()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Section 3: Cluster Hyperparameters
    max_allowed_k = min(10, len(X_clean) - 1)
    if max_allowed_k < 2:
        st.error("❌ Dataset does not have enough rows to form at least 2 clusters.")
        st.stop()

    default_k = 5 if 5 <= max_allowed_k else max_allowed_k

    st.markdown("""
    <div class="sidebar-card-header">
        <span>🎯 Model Hyperparameters</span>
        <span style="color: #FBBF24; font-size: 0.7rem;">Lloyd k-means++</span>
    </div>
    """, unsafe_allow_html=True)

    k_slider_key = f"k_slider_{active_data_label}_{max_allowed_k}"
    selected_k = st.slider(
        "Number of Clusters (K):",
        min_value=2,
        max_value=max_allowed_k,
        value=default_k,
        key=k_slider_key,
        help="Select the number of centroids (K) for K-Means."
    )

    random_state = st.number_input(
        "Random State Seed:",
        min_value=0,
        max_value=9999,
        value=0,
        step=1,
        help="Keeps K-Means initialization deterministic and reproducible."
    )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Section 4: Live Diagnostics Card
    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-header">
            <span>📊 Live Telemetry</span>
            <span style="color: #34D399;">● Online</span>
        </div>
        <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.7;">
            <div>📁 <b>Source:</b> <span style="color: #93C5FD;">{active_data_label.split(' ')[0]}</span></div>
            <div>👥 <b>Sample Size:</b> {len(df):,} rows</div>
            <div>✨ <b>Valid Feature Rows:</b> {len(X_clean):,}</div>
            <div>📐 <b>Dimensions:</b> {len(selected_features)} features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Run Model & Compute Metrics
# ---------------------------------------------------------
model, labels, sil_score, inertia = fit_kmeans_model(
    X_clean, n_clusters=selected_k, random_state=random_state
)

# Assign cluster column safely
df_clustered = df.copy()
df_clustered.loc[valid_mask, "Cluster"] = labels
df_clustered["Cluster"] = df_clustered["Cluster"].fillna(-1).astype(int)


# ---------------------------------------------------------
# Main App Header & Meta Bar
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span>✦</span> COLLEGE DWM MINI-PROJECT &bull; CUSTOMER SEGMENTATION ENGINE
    </div>
    <div class="hero-title">Customer Segmentation Studio</div>
    <div class="hero-subtitle">
        Uncover high-value customer cohorts using unsupervised <b>K-Means Clustering</b>. 
        Analyze cluster dispersion via the <b>Elbow Method</b> and <b>Silhouette Cohesion</b>, and deploy tailored, data-backed marketing initiatives.
    </div>
</div>
""", unsafe_allow_html=True)

# Interactive Meta Chip Bar
st.markdown(f"""
<div class="meta-bar">
    <div class="meta-chip">📁 Source: <strong>{active_data_label}</strong></div>
    <div class="meta-chip">⚙️ Algorithm: <strong>Lloyd K-Means++</strong></div>
    <div class="meta-chip">📐 Feature Pair: <strong>{selected_features[0]} &times; {selected_features[1]}</strong></div>
    <div class="meta-chip">🎯 Clusters Formed: <strong>K = {selected_k}</strong></div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# KPI Metric Cards Grid
# ---------------------------------------------------------
sil_display = f"{sil_score:.3f}" if sil_score is not None else "N/A"
if sil_score is not None:
    if sil_score >= 0.50:
        sil_badge_html = '<span class="kpi-pill kpi-pill-emerald">● Strong Cohesion</span>'
    elif sil_score >= 0.30:
        sil_badge_html = '<span class="kpi-pill kpi-pill-indigo">● Moderate Separation</span>'
    else:
        sil_badge_html = '<span class="kpi-pill kpi-pill-amber">● Weak Structure</span>'
else:
    sil_badge_html = '<span class="kpi-pill kpi-pill-amber">● Single Cluster</span>'

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">Customers Analyzed</span>
            <div class="kpi-icon-badge">👥</div>
        </div>
        <div class="kpi-value">{len(X_clean):,}</div>
        <div class="kpi-sub">
            <span class="kpi-pill kpi-pill-emerald">100% Valid</span>
            <span>{len(df) - len(X_clean)} excluded</span>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">Active Clusters (K)</span>
            <div class="kpi-icon-badge">🎯</div>
        </div>
        <div class="kpi-value">{selected_k}</div>
        <div class="kpi-sub">
            <span class="kpi-pill kpi-pill-indigo">Selected</span>
            <span>Max limit: {max_allowed_k}</span>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">Silhouette Score</span>
            <div class="kpi-icon-badge">⚡</div>
        </div>
        <div class="kpi-value">{sil_display}</div>
        <div class="kpi-sub">
            {sil_badge_html}
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">WCSS (Inertia)</span>
            <div class="kpi-icon-badge">📉</div>
        </div>
        <div class="kpi-value">{inertia:,.0f}</div>
        <div class="kpi-sub">
            <span class="kpi-pill kpi-pill-indigo">Compactness</span>
            <span>Sum of squared errors</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Tabs: Organized Dashboard Views
# ---------------------------------------------------------
tab_viz, tab_report, tab_elbow, tab_data = st.tabs([
    "🎨 Cluster Visualization",
    "📋 Segment Personas & Playbook",
    "📈 Optimal Clusters (Elbow Method)",
    "📄 Dataset Exploration"
])

# Vibrant modern color palette for clusters
CLUSTER_COLORS = [
    "#38BDF8",  # Neon Sky
    "#818CF8",  # Indigo
    "#34D399",  # Emerald
    "#F472B6",  # Rose Pink
    "#FBBF24",  # Amber
    "#A78BFA",  # Violet
    "#FB7185",  # Coral
    "#2DD4BF",  # Teal
    "#F97316",  # Orange
    "#06B6D4"   # Cyan
]

# ---------------------------------------------------------
# TAB 1: Cluster Visualization
# ---------------------------------------------------------
with tab_viz:
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 12px;">
        <div>
            <h3 style="margin-bottom: 2px;">🎨 2D Customer Cluster Distribution</h3>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 0;">
                Showing customers grouped along <b>{selected_features[0]}</b> vs <b>{selected_features[1]}</b>.
                Yellow stars (<span style="color: #FBBF24;">⭐</span>) highlight mathematical cluster centroids.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=120)
    apply_dark_theme(fig, ax)

    # Plot customer points by cluster
    for c_id in range(selected_k):
        c_mask = (df_clustered["Cluster"] == c_id)
        ax.scatter(
            df_clustered.loc[c_mask, selected_features[0]],
            df_clustered.loc[c_mask, selected_features[1]],
            s=85,
            color=CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)],
            alpha=0.88,
            edgecolors='#FFFFFF',
            linewidths=0.6,
            label=f"Cluster {c_id}"
        )

    # Plot glowing centroids
    centroids = model.cluster_centers_
    # Outer glow
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=480,
        color="#FBBF24",
        alpha=0.35,
        marker="o",
        zorder=9
    )
    # Centroid star
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=320,
        color="#FDE047",
        edgecolors="#000000",
        linewidths=1.8,
        marker="*",
        zorder=10,
        label="Centroids (★)"
    )

    ax.set_title(
        f"K-Means Clustering (K={selected_k}) with Centroid Coordinates",
        fontsize=13,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel(selected_features[0], fontsize=10.5, fontweight="600")
    ax.set_ylabel(selected_features[1], fontsize=10.5, fontweight="600")
    
    legend = ax.legend(
        title="Customer Cohorts",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
        facecolor="#111827",
        edgecolor="#334155",
        labelcolor="#E2E8F0"
    )
    legend.get_title().set_color("#F8FAFC")
    legend.get_title().set_fontweight("bold")
    
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Centroid Coordinates Quick Bar
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        st.markdown("""
        <div class="info-callout">
            💡 <b>Centroid Anatomy:</b> Each star marks the multidimensional center of mass for that cluster. 
            Customers closest to a centroid exhibit the truest behavioral signature of that consumer segment.
        </div>
        """, unsafe_allow_html=True)
    
    with col_c2:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        csv_data = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Clustered Dataset (CSV)",
            data=csv_data,
            file_name=f"customer_clusters_k{selected_k}.csv",
            mime="text/csv",
            width="stretch"
        )


# ---------------------------------------------------------
# TAB 2: Cluster Personas & Strategic Playbook
# ---------------------------------------------------------
with tab_report:
    st.markdown("### 📋 Customer Archetypes & Strategic Marketing Playbook")
    st.write(
        "Each segment is synthesized into a business persona based on its distance from the mean "
        "of the selected feature pair:"
    )

    summary_df, insights = generate_cluster_insights(
        df_clustered[df_clustered["Cluster"] >= 0],
        selected_features,
        labels,
        k=selected_k
    )

    # Clean Styled Dataframe
    st.dataframe(
        summary_df.style.format({
            "Share (%)": "{:.1f}%",
            f"Avg {selected_features[0]}": "{:.2f}",
            f"Avg {selected_features[1]}": "{:.2f}"
        }),
        width="stretch"
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Executive Persona Cards
    for item in insights:
        c_id = item["cluster_id"]
        c_color = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]

        st.markdown(f"""
        <div class="persona-card" style="border-left-color: {c_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.25rem; font-weight: 700; color: {c_color};">
                        Cluster {c_id}: {item["archetype"]}
                    </span>
                </div>
                <span class="persona-badge" style="border-color: {c_color}55; color: {c_color};">
                    {item["count"]} Customers &bull; {item["pct"]}% of Total
                </span>
            </div>
            <div style="font-size: 0.92rem; color: #CBD5E1; margin-bottom: 10px; line-height: 1.5;">
                <b>Profile:</b> {item["tier1"]} <i>{selected_features[0]}</i> (mean: <b>{item["f1_mean"]}</b>), 
                {item["tier2"]} <i>{selected_features[1]}</i> (mean: <b>{item["f2_mean"]}</b>).<br>
                {item["description"]}
            </div>
            <div class="persona-strategy-box">
                🎯 <strong style="color: #F8FAFC;">Recommended Action:</strong> 
                <span style="color: #CBD5E1;">{item["recommendation"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 3: Elbow & Silhouette Optimization
# ---------------------------------------------------------
with tab_elbow:
    st.markdown("### 📈 Determining Optimal Clusters: Elbow & Silhouette Criteria")
    st.write(
        "To select the best number of clusters (K), data scientists balance **Within-Cluster Sum of Squares (Inertia)** "
        "and the **Silhouette Coefficient** (measuring how distinct and well-separated clusters are from one another)."
    )

    with st.spinner("Computing WCSS & Silhouette across K = 1 to 10..."):
        elbow_data = calculate_elbow(X_clean, max_k=10, random_state=random_state)

    fig_elbow, (ax_wcss, ax_sil) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=120)
    apply_dark_theme(fig_elbow, [ax_wcss, ax_sil])

    # Left: WCSS Inertia Plot
    ax_wcss.plot(
        elbow_data['k_range'], elbow_data['wcss'],
        marker='o', linewidth=2.4, color='#38BDF8', markersize=7
    )
    if selected_k in elbow_data['k_range']:
        k_idx = elbow_data['k_range'].index(selected_k)
        ax_wcss.scatter(
            selected_k, elbow_data['wcss'][k_idx],
            color='#EF4444', s=220, zorder=6, edgecolors='#FFFFFF', linewidths=1.5,
            label=f"Active K={selected_k}"
        )
    ax_wcss.set_title("WCSS (Inertia) vs K (Elbow Curve)", fontsize=11.5, fontweight="bold", pad=10)
    ax_wcss.set_xlabel("Number of Clusters (K)", fontsize=9.5, fontweight="600")
    ax_wcss.set_ylabel("Within-Cluster Sum of Squares", fontsize=9.5, fontweight="600")
    ax_wcss.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#E2E8F0")

    # Right: Silhouette Score Plot
    k_vals_sil = [k for k, s in zip(elbow_data['k_range'], elbow_data['silhouette']) if s is not None]
    sils_valid = [s for s in elbow_data['silhouette'] if s is not None]
    
    if sils_valid:
        ax_sil.plot(
            k_vals_sil, sils_valid,
            marker='s', linewidth=2.4, color='#34D399', markersize=7
        )
        if selected_k in k_vals_sil:
            sil_val = elbow_data['silhouette'][elbow_data['k_range'].index(selected_k)]
            ax_sil.scatter(
                selected_k, sil_val,
                color='#EF4444', s=220, zorder=6, edgecolors='#FFFFFF', linewidths=1.5,
                label=f"Active K={selected_k} ({sil_val})"
            )
        ax_sil.set_title("Silhouette Score vs K (Higher = Better)", fontsize=11.5, fontweight="bold", pad=10)
        ax_sil.set_xlabel("Number of Clusters (K)", fontsize=9.5, fontweight="600")
        ax_sil.set_ylabel("Silhouette Score", fontsize=9.5, fontweight="600")
        ax_sil.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#E2E8F0")

    st.pyplot(fig_elbow, width="stretch")
    plt.close(fig_elbow)

    opt_k_sil = elbow_data['optimal_k_by_silhouette']
    opt_score = elbow_data['silhouette'][elbow_data['k_range'].index(opt_k_sil)] if opt_k_sil in elbow_data['k_range'] else 'N/A'
    
    st.markdown(f"""
    <div class="info-callout">
        🔍 <b>Mathematical Analysis:</b> The primary elbow bend for <b>{selected_features[0]}</b> vs <b>{selected_features[1]}</b> 
        occurs sharply around <b>K = {opt_k_sil}</b>. This configuration maximizes cluster separation with a peak Silhouette Score of 
        <b style="color: #34D399;">{opt_score}</b>. Choosing values beyond this yields diminishing returns and fragments natural consumer groups.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 4: Dataset Exploration
# ---------------------------------------------------------
with tab_data:
    st.markdown("### 📄 Raw Data Exploration & Feature Distribution")
    
    col_d1, col_d2 = st.columns([3, 2])
    with col_d1:
        st.markdown("**Sample Records (First 10 Rows):**")
        st.dataframe(df.head(10), width="stretch")
    
    with col_d2:
        st.markdown("**Dataset Health & Structure:**")
        info_df = pd.DataFrame([
            {"Metric": "Total Rows", "Value": f"{len(df):,}"},
            {"Metric": "Total Columns", "Value": str(len(df.columns))},
            {"Metric": "Identified Numeric Columns", "Value": str(len(numeric_columns))},
            {"Metric": "Missing / Null Values", "Value": str(df.isnull().sum().sum())},
            {"Metric": "Duplicate Records", "Value": str(df.duplicated().sum())}
        ])
        st.dataframe(info_df, width="stretch", hide_index=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("**Descriptive Statistics for Numeric Features:**")
    st.dataframe(df.describe().T.style.format("{:.2f}"), width="stretch")

st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 16px; display: flex; justify-content: space-between; align-items: center; color: #64748B; font-size: 0.8rem;">
    <div>College Data Warehousing & Mining (DWM) Mini-Project &bull; SegmentIQ Studio</div>
    <div>Powered by Scikit-Learn & Streamlit</div>
</div>
""", unsafe_allow_html=True)
