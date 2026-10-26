"""
clustering_helper.py
Core data processing, K-Means clustering, and statistical profiling utilities
for the Customer Segmentation project.
"""

import io
import re
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_dataset(source):
    """
    Safely load a dataset from a file path, file-like object, or BytesIO.
    Returns: (df, error_message)
    """
    if source is None:
        return None, "No file provided."
    
    try:
        if isinstance(source, (str, io.BytesIO, io.StringIO)) or hasattr(source, "read"):
            df = pd.read_csv(source)
        else:
            return None, "Unsupported file format or source type."
            
        if df.empty:
            return None, "The uploaded CSV file is empty."
            
        if len(df) < 2:
            return None, "Dataset must have at least 2 rows for clustering."
            
        return df, None
    except pd.errors.EmptyDataError:
        return None, "The CSV file is completely empty."
    except pd.errors.ParserError as e:
        return None, f"Failed to parse CSV file: {e}"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def get_numeric_columns(df, exclude_id=True):
    """
    Identify numeric columns suitable for clustering.
    Optionally separates out identifier columns (e.g. 'CustomerID').
    """
    if df is None or df.empty:
        return []
    
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    
    if not exclude_id:
        return numeric_cols
        
    # Heuristic to detect ID columns
    id_pattern = re.compile(r'^(customer[-_\s]?id|id|sr[-_\s]?no|sno|serial)$', re.IGNORECASE)
    filtered = [col for col in numeric_cols if not id_pattern.match(col.strip())]
    
    # If filtering removed everything, fall back to all numeric columns
    return filtered if len(filtered) >= 2 else numeric_cols


def get_safe_default_features(numeric_cols, preferred=("Annual Income (k$)", "Spending Score (1-100)")):
    """
    Safely return default feature names that are GUARANTEED to exist in numeric_cols.
    Prevents StreamlitDefaultNotInOptionsError.
    """
    if not numeric_cols:
        return []
        
    # Check if both preferred columns exist
    if all(pref in numeric_cols for pref in preferred):
        return list(preferred)
        
    # Otherwise select up to first 2 numeric columns
    return numeric_cols[:min(2, len(numeric_cols))]


def prepare_feature_matrix(df, features):
    """
    Extract and validate the numerical feature matrix X.
    Handles missing values and inf values safely.
    Returns: (X_clean, valid_mask, error_msg)
    """
    if not features or len(features) < 2:
        return None, None, "Please select at least 2 features."
        
    for feat in features:
        if feat not in df.columns:
            return None, None, f"Feature '{feat}' not found in dataset."
            
    X = df[features].copy()
    
    # Check for NaN / infinite values
    valid_mask = ~X.isna().any(axis=1) & ~np.isinf(X).any(axis=1)
    X_clean = X[valid_mask]
    
    if len(X_clean) < 2:
        return None, None, "Insufficient valid numeric rows after removing nulls/infinities."
        
    return X_clean, valid_mask, None


def calculate_elbow(X, max_k=10, random_state=0):
    """
    Calculate WCSS (inertia) and silhouette scores for K from 1 to max_k.
    Returns: dict with 'k_range', 'wcss', 'silhouette', 'optimal_k'
    """
    n_samples = len(X)
    k_upper = min(max_k, n_samples - 1)
    if k_upper < 1:
        k_upper = 1
        
    k_range = list(range(1, k_upper + 1))
    wcss = []
    sil_scores = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        
        if k > 1 and n_samples > k:
            score = silhouette_score(X, labels)
            sil_scores.append(round(score, 4))
        else:
            sil_scores.append(None)
            
    # Find best silhouette score among k >= 2
    valid_sils = [(k, s) for k, s in zip(k_range, sil_scores) if s is not None]
    optimal_k = max(valid_sils, key=lambda x: x[1])[0] if valid_sils else (5 if 5 in k_range else k_range[0])
    
    return {
        'k_range': k_range,
        'wcss': wcss,
        'silhouette': sil_scores,
        'optimal_k_by_silhouette': optimal_k
    }


def fit_kmeans_model(X, n_clusters=5, random_state=0):
    """
    Fit KMeans model on feature matrix X.
    Returns: (model, labels, silhouette, inertia)
    """
    if n_clusters > len(X):
        raise ValueError(f"Clusters ({n_clusters}) cannot exceed sample size ({len(X)}).")
        
    model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    
    sil = None
    if n_clusters > 1 and len(X) > n_clusters:
        sil = round(silhouette_score(X, labels), 4)
        
    return model, labels, sil, round(model.inertia_, 2)


def get_feature_tier(val, mean_val, std_val):
    """
    Categorize a mean value into Low, Moderate, or High using a +/- 0.35 std band.
    """
    if std_val == 0 or np.isnan(std_val):
        return "Moderate"
    low_bound = mean_val - 0.35 * std_val
    high_bound = mean_val + 0.35 * std_val
    
    if val < low_bound:
        return "Low"
    elif val > high_bound:
        return "High"
    return "Moderate"


def generate_cluster_insights(df, features, labels, k):
    """
    Compute cluster summary table and tailored business recommendations.
    Returns: (summary_df, insights_list)
    """
    f1, f2 = features[0], features[1]
    
    # Calculate dataset-wide statistics
    m1, s1 = df[f1].mean(), df[f1].std()
    m2, s2 = df[f2].mean(), df[f2].std()
    
    summary_data = []
    insights = []
    total_customers = len(df)
    
    for c in range(k):
        c_mask = (labels == c)
        c_df = df[c_mask]
        count = len(c_df)
        pct = (count / total_customers) * 100 if total_customers > 0 else 0
        
        c_m1 = c_df[f1].mean() if count > 0 else 0
        c_m2 = c_df[f2].mean() if count > 0 else 0
        
        tier1 = get_feature_tier(c_m1, m1, s1)
        tier2 = get_feature_tier(c_m2, m2, s2)
        
        # Determine business archetype based on the two tiers
        archetype, description, recommendation = get_business_archetype(
            f1, f2, tier1, tier2, c_m1, c_m2
        )
        
        summary_data.append({
            "Cluster ID": c,
            "Customer Count": count,
            "Share (%)": round(pct, 1),
            f"Avg {f1}": round(c_m1, 2),
            f"Avg {f2}": round(c_m2, 2),
            f"{f1} Tier": tier1,
            f"{f2} Tier": tier2,
            "Segment Archetype": archetype
        })
        
        insights.append({
            "cluster_id": c,
            "count": count,
            "pct": round(pct, 1),
            "tier1": tier1,
            "tier2": tier2,
            "f1_mean": round(c_m1, 2),
            "f2_mean": round(c_m2, 2),
            "archetype": archetype,
            "description": description,
            "recommendation": recommendation
        })
        
    return pd.DataFrame(summary_data), insights


def get_business_archetype(f1, f2, tier1, tier2, val1, val2):
    """
    Produce data-grounded archetype naming, description, and marketing action.
    """
    # High Income & High Spending
    if tier1 == "High" and tier2 == "High":
        archetype = "VIP / High-Value Spenders"
        desc = f"Affluent customers with high {f1} and high {f2}. They represent the highest lifetime customer value."
        rec = "Offer luxury exclusives, VIP loyalty perks, personalized concierge service, and early access to premium collections."
    # High Income & Low Spending
    elif tier1 == "High" and tier2 == "Low":
        archetype = "Conservative Affluent"
        desc = f"Customers with high {f1} but cautious/low {f2}. They have substantial spending potential that is currently untapped."
        rec = "Engage with premium value propositions, quality guarantees, tailored invitations, and targeted seasonal promotions."
    # Low Income & High Spending
    elif tier1 == "Low" and tier2 == "High":
        archetype = "Aspirational / Trend Spenders"
        desc = f"Customers with lower {f1} but high {f2}. Likely trend-conscious or impulse shoppers who prioritize lifestyle purchases."
        rec = "Promote trendy products, flash sales, installment plans (BNPL), and vibrant social media influencer campaigns."
    # Low Income & Low Spending
    elif tier1 == "Low" and tier2 == "Low":
        archetype = "Budget-Conscious / Economical"
        desc = f"Customers with lower {f1} and disciplined/low {f2}. They prioritize affordability and essential needs."
        rec = "Focus on discount coupons, bundle packages, value-for-money deals, and free delivery incentives."
    # Moderate Income & Moderate Spending
    elif tier1 == "Moderate" and tier2 == "Moderate":
        archetype = "Mainstream / Core Regulars"
        desc = f"Balanced customers with moderate {f1} and moderate {f2}. They form the consistent backbone of store traffic."
        rec = "Nurture with points-based loyalty programs, seasonal newsletters, and cross-category discovery discounts."
    # Mixed / Intermediate cases
    elif tier1 == "Moderate" and tier2 == "High":
        archetype = "Active Lifestyle Spenders"
        desc = f"Moderate {f1} with eager {f2}. They actively spend on experiences and high-affinity products."
        rec = "Target with dynamic bundle offers, weekend promotions, and repeat-buyer rewards."
    elif tier1 == "Moderate" and tier2 == "Low":
        archetype = "Selective Savers"
        desc = f"Moderate {f1} with conservative {f2}. They spend selectively on essentials or proven items."
        rec = "Highlight practical utility, customer reviews, and risk-free satisfaction guarantees."
    elif tier1 == "High" and tier2 == "Moderate":
        archetype = "Stable Affluent"
        desc = f"High {f1} with steady, balanced {f2}. Reliable buyers with high financial security."
        rec = "Encourage upselling into premium lines with personal shopper recommendations."
    else:  # Low tier1, Moderate tier2
        archetype = "Emerging Practical Spenders"
        desc = f"Lower {f1} with moderate {f2}. Pragmatic buyers managing their spending thoughtfully."
        rec = "Provide competitive pricing on popular items and festive discount vouchers."

    return archetype, desc, rec
