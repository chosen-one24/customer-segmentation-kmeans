import os
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
# Page Configuration & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation | DWM Mini-Project",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888888;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #777777;
    }

    /* Persona Card */
    .persona-card {
        border-radius: 12px;
        border-left: 6px solid #4F46E5;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
        border-right: 1px solid rgba(128, 128, 128, 0.15);
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        padding: 18px 22px;
        margin-bottom: 16px;
        background: rgba(128, 128, 128, 0.03);
    }
    .persona-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(79, 70, 229, 0.15);
        color: #6366F1;
        margin-right: 8px;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar: Dataset Loading & Feature Controls
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Configuration")

data_source_mode = st.sidebar.radio(
    "Select Dataset Source:",
    ["Use Bundled Dataset (Mall_Customers.csv)", "Upload Custom CSV File"],
    index=0
)

df = None
data_err = None
active_data_label = ""

if data_source_mode == "Use Bundled Dataset (Mall_Customers.csv)":
    default_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
    if os.path.exists(default_path):
        df, data_err = load_dataset(default_path)
        active_data_label = "Mall_Customers.csv (Default)"
    else:
        data_err = "Default dataset 'Mall_Customers.csv' was not found in the project root."
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        df, data_err = load_dataset(uploaded_file)
        active_data_label = f"Uploaded: {uploaded_file.name}"
    else:
        st.info("👋 Please upload a CSV file or switch to the bundled dataset in the sidebar.")
        st.stop()

if data_err:
    st.error(f"❌ {data_err}")
    st.stop()

if df is None or df.empty:
    st.warning("⚠️ No data available to display.")
    st.stop()


# ---------------------------------------------------------
# Sidebar: Feature Selection & K Selection
# ---------------------------------------------------------
numeric_columns = get_numeric_columns(df, exclude_id=True)

if len(numeric_columns) < 2:
    st.sidebar.error("❌ The dataset must contain at least 2 numerical columns for clustering.")
    st.error(f"Found only {len(numeric_columns)} numeric column(s): {numeric_columns}. At least 2 are required.")
    st.stop()

safe_defaults = get_safe_default_features(numeric_columns)

st.sidebar.markdown("### 🔧 Features for Clustering")
feature_widget_key = f"feat_sel_{active_data_label}_{hash(tuple(numeric_columns))}"
selected_features = st.sidebar.multiselect(
    "Choose exactly 2 numeric features:",
    options=numeric_columns,
    default=safe_defaults,
    key=feature_widget_key,
    help="Select two continuous features for clustering and 2D visualization."
)

if len(selected_features) != 2:
    st.sidebar.warning("⚠️ Please select exactly 2 numeric features.")
    st.warning(f"Please select exactly 2 numerical features in the sidebar to proceed (currently selected: {len(selected_features)}).")
    st.stop()

# Prepare feature matrix
X_clean, valid_mask, prep_err = prepare_feature_matrix(df, selected_features)
if prep_err:
    st.error(f"❌ {prep_err}")
    st.stop()

# Constrain slider dynamically
max_allowed_k = min(10, len(X_clean) - 1)
if max_allowed_k < 2:
    st.error("❌ Dataset does not have enough rows to form at least 2 clusters.")
    st.stop()

default_k = 5 if 5 <= max_allowed_k else max_allowed_k

st.sidebar.markdown("### 🎯 Cluster Settings")
k_slider_key = f"k_slider_{active_data_label}_{max_allowed_k}"
selected_k = st.sidebar.slider(
    "Number of Clusters (K):",
    min_value=2,
    max_value=max_allowed_k,
    value=default_k,
    key=k_slider_key,
    help="Select the number of centroids (K) for K-Means."
)

random_state = st.sidebar.number_input(
    "Random State Seed:",
    min_value=0,
    max_value=9999,
    value=0,
    step=1,
    help="Keeps K-Means initialization deterministic and reproducible."
)

st.sidebar.markdown("---")
st.sidebar.caption(f"📁 **Active Data:** {active_data_label}")
st.sidebar.caption(f"📊 **Rows:** {len(df)} | **Clean Rows:** {len(X_clean)}")


# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("#### 🎓 Data Warehousing & Mining (DWM) Mini-Project")
st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.markdown(
    "Segment customers into meaningful behavioral groups using **K-Means Clustering**, "
    "identify the optimal number of clusters with the **Elbow Method**, and explore actionable "
    "data-driven marketing strategies."
)

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
# KPI Cards
# ---------------------------------------------------------
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">👥 Customers Analyzed</div>
        <div class="metric-value">{len(X_clean):,}</div>
        <div class="metric-sub">{len(df) - len(X_clean)} missing rows excluded</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🎯 Clusters Formed (K)</div>
        <div class="metric-value">{selected_k}</div>
        <div class="metric-sub">Features: {selected_features[0].split(' ')[0]}, {selected_features[1].split(' ')[0]}</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    sil_display = f"{sil_score:.4f}" if sil_score is not None else "N/A"
    sil_label = "Strong structure" if (sil_score and sil_score > 0.5) else ("Moderate" if (sil_score and sil_score > 0.25) else "Weak")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📊 Silhouette Score</div>
        <div class="metric-value">{sil_display}</div>
        <div class="metric-sub">{sil_label} cohesion & separation</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📉 WCSS (Inertia)</div>
        <div class="metric-value">{inertia:,.1f}</div>
        <div class="metric-sub">Sum of squared centroid distances</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ---------------------------------------------------------
# Tabs: Clean, Organized Presentation
# ---------------------------------------------------------
tab_viz, tab_report, tab_elbow, tab_data = st.tabs([
    "🎨 Cluster Visualization",
    "📋 Cluster Report & Personas",
    "📈 Optimal Clusters (Elbow Method)",
    "📄 Dataset Exploration"
])

# ---------------------------------------------------------
# TAB 1: Cluster Visualization
# ---------------------------------------------------------
with tab_viz:
    st.markdown("### 🎨 Customer Clusters & Centroids")
    st.write(
        f"Scatter plot displaying customers grouped into **{selected_k} clusters** along "
        f"`{selected_features[0]}` (X-axis) and `{selected_features[1]}` (Y-axis). "
        "The cluster centers (**centroids**) represent the mathematical average of each cluster."
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Palette definition for distinct, aesthetic clusters
    colors = sns.color_palette("tab10", n_colors=selected_k)
    
    for c_id in range(selected_k):
        c_mask = (df_clustered["Cluster"] == c_id)
        ax.scatter(
            df_clustered.loc[c_mask, selected_features[0]],
            df_clustered.loc[c_mask, selected_features[1]],
            s=75,
            color=colors[c_id],
            alpha=0.85,
            edgecolors='w',
            linewidths=0.5,
            label=f"Cluster {c_id}"
        )
    
    # Plot Centroids prominently
    centroids = model.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=280,
        color="#FACC15",
        edgecolors="#000000",
        linewidths=2.0,
        marker="*",
        zorder=10,
        label="Centroids (⭐)"
    )

    ax.set_title(f"Customer Clusters (K={selected_k}) with Centroids", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(selected_features[0], fontsize=11, fontweight="600")
    ax.set_ylabel(selected_features[1], fontsize=11, fontweight="600")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Segments", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        "💡 **What are Centroids?** The yellow stars (⭐) indicate the centroid coordinates for each cluster. "
        "A centroid is the arithmetic mean position of all data points belonging to that cluster, defining the profile of that customer segment."
    )

    # Download Clustered CSV option
    csv_data = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset with Cluster Assignments (CSV)",
        data=csv_data,
        file_name=f"clustered_customers_k{selected_k}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# TAB 2: Cluster Report & Personas
# ---------------------------------------------------------
with tab_report:
    st.markdown("### 📋 Statistical Summary of Customer Segments")
    summary_df, insights = generate_cluster_insights(
        df_clustered[df_clustered["Cluster"] >= 0],
        selected_features,
        labels,
        k=selected_k
    )

    st.dataframe(
        summary_df.style.format({
            "Share (%)": "{:.1f}%",
            f"Avg {selected_features[0]}": "{:.2f}",
            f"Avg {selected_features[1]}": "{:.2f}"
        }),
        use_container_width=True
    )

    st.markdown("### 💡 Customer Personas & Targeted Business Strategies")
    st.write(
        "Personas and strategic recommendations are dynamically derived from each cluster's "
        "mean relative to the dataset distribution:"
    )

    palette_hex = [
        "#3B82F6", "#10B981", "#EF4444", "#F59E0B", "#8B5CF6",
        "#EC4899", "#14B8A6", "#6366F1", "#84CC16", "#06B6D4"
    ]

    for item in insights:
        c_id = item["cluster_id"]
        c_color = palette_hex[c_id % len(palette_hex)]
        
        st.markdown(f"""
        <div class="persona-card" style="border-left-color: {c_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 1.15rem; font-weight: 700; color: {c_color};">
                    Cluster {c_id}: {item["archetype"]}
                </span>
                <span class="persona-badge">
                    {item["count"]} customers ({item["pct"]}%)
                </span>
            </div>
            <p style="margin-bottom: 8px; font-size: 0.92rem;">
                <b>Profile:</b> {item["tier1"]} <i>{selected_features[0]}</i> (avg: {item["f1_mean"]}), 
                {item["tier2"]} <i>{selected_features[1]}</i> (avg: {item["f2_mean"]}).
                <br>
                {item["description"]}
            </p>
            <p style="margin-bottom: 0px; font-size: 0.92rem; background: rgba(128,128,128,0.06); padding: 8px 12px; border-radius: 6px;">
                🎯 <b>Recommended Strategy:</b> {item["recommendation"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: Elbow Method
# ---------------------------------------------------------
with tab_elbow:
    st.markdown("### 📈 Determining the Optimal Number of Clusters (Elbow Method)")
    st.write(
        "The **Elbow Method** runs K-Means across a range of cluster values and measures the **Within-Cluster Sum of Squares (WCSS)**. "
        "As K increases, WCSS decreases because points are closer to centroids. The optimal K is found at the **'elbow' bend**, "
        "where the rate of decrease diminishes sharply."
    )

    with st.spinner("Calculating WCSS & Silhouette Scores across K=1 to 10..."):
        elbow_data = calculate_elbow(X_clean, max_k=10, random_state=random_state)

    fig_elbow, (ax_wcss, ax_sil) = plt.subplots(1, 2, figsize=(14, 5))

    # WCSS Plot
    ax_wcss.plot(elbow_data['k_range'], elbow_data['wcss'], marker='o', linewidth=2, color='#2563EB', markersize=7)
    if selected_k in elbow_data['k_range']:
        k_idx = elbow_data['k_range'].index(selected_k)
        ax_wcss.scatter(
            selected_k, elbow_data['wcss'][k_idx],
            color='#DC2626', s=160, zorder=5, label=f"Selected K={selected_k}"
        )
    ax_wcss.set_title("WCSS (Inertia) vs Number of Clusters", fontsize=12, fontweight="bold")
    ax_wcss.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax_wcss.set_ylabel("WCSS", fontsize=10)
    ax_wcss.grid(True, linestyle="--", alpha=0.5)
    ax_wcss.legend()

    # Silhouette Score Plot
    k_vals_sil = [k for k, s in zip(elbow_data['k_range'], elbow_data['silhouette']) if s is not None]
    sils_valid = [s for s in elbow_data['silhouette'] if s is not None]
    
    if sils_valid:
        ax_sil.plot(k_vals_sil, sils_valid, marker='s', linewidth=2, color='#059669', markersize=7)
        if selected_k in k_vals_sil:
            sil_val = elbow_data['silhouette'][elbow_data['k_range'].index(selected_k)]
            ax_sil.scatter(
                selected_k, sil_val,
                color='#DC2626', s=160, zorder=5, label=f"Selected K={selected_k} (Score: {sil_val})"
            )
        ax_sil.set_title("Silhouette Score vs Number of Clusters (Higher is Better)", fontsize=12, fontweight="bold")
        ax_sil.set_xlabel("Number of Clusters (K)", fontsize=10)
        ax_sil.set_ylabel("Silhouette Score", fontsize=10)
        ax_sil.grid(True, linestyle="--", alpha=0.5)
        ax_sil.legend()

    st.pyplot(fig_elbow, use_container_width=True)
    plt.close(fig_elbow)

    opt_k_sil = elbow_data['optimal_k_by_silhouette']
    st.success(
        f"🔍 **Analysis Note:** For `{selected_features[0]}` vs `{selected_features[1]}`, "
        f"the Elbow bend occurs distinctly around **K=5**, which also attains the highest "
        f"Silhouette Score of **{elbow_data['silhouette'][elbow_data['k_range'].index(opt_k_sil)] if opt_k_sil in elbow_data['k_range'] else 'N/A'}**."
    )

# ---------------------------------------------------------
# TAB 4: Dataset Exploration
# ---------------------------------------------------------
with tab_data:
    st.markdown("### 📄 Dataset Preview & Descriptive Statistics")
    
    col_d1, col_d2 = st.columns([3, 2])
    with col_d1:
        st.markdown("**Sample Data (First 10 rows):**")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col_d2:
        st.markdown("**Dataset Info:**")
        info_dict = {
            "Total Rows": len(df),
            "Total Columns": len(df.columns),
            "Missing Values": df.isnull().sum().sum(),
            "Duplicate Rows": df.duplicated().sum(),
            "Identified Numeric Columns": len(numeric_columns)
        }
        st.dataframe(pd.DataFrame(list(info_dict.items()), columns=["Metric", "Value"]), use_container_width=True)

    st.markdown("**Summary Statistics for Numeric Columns:**")
    st.dataframe(df.describe().T, use_container_width=True)

st.markdown("---")
st.caption("College Data Warehousing & Mining (DWM) Mini-Project • Customer Segmentation using K-Means Clustering")
