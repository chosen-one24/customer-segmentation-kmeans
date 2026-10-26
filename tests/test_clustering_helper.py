import os
import io
import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clustering_helper import (
    load_dataset,
    get_numeric_columns,
    get_safe_default_features,
    prepare_feature_matrix,
    calculate_elbow,
    fit_kmeans_model,
    generate_cluster_insights,
    get_feature_tier
)

def test_load_dataset_valid():
    df, err = load_dataset('Mall_Customers.csv')
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert 'Annual Income (k$)' in df.columns
    assert 'Spending Score (1-100)' in df.columns

def test_load_dataset_empty_file():
    empty_stream = io.StringIO("")
    df, err = load_dataset(empty_stream)
    assert df is None
    assert "empty" in err.lower()

def test_load_dataset_none():
    df, err = load_dataset(None)
    assert df is None
    assert "no file" in err.lower()

def test_load_dataset_too_few_rows():
    tiny_csv = io.StringIO("A,B\n1,2\n")
    df, err = load_dataset(tiny_csv)
    assert df is None
    assert "at least 2 rows" in err

def test_get_numeric_columns():
    df = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Gender': ['Male', 'Female', 'Female'],
        'Age': [20, 30, 40],
        'Score': [50, 60, 70]
    })
    cols = get_numeric_columns(df, exclude_id=True)
    assert 'Age' in cols
    assert 'Score' in cols
    assert 'CustomerID' not in cols
    assert 'Gender' not in cols

def test_get_safe_default_features_when_preferred_present():
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    defaults = get_safe_default_features(numeric_cols)
    assert defaults == ['Annual Income (k$)', 'Spending Score (1-100)']

def test_get_safe_default_features_when_preferred_absent():
    numeric_cols = ['Feature_A', 'Feature_B', 'Feature_C']
    defaults = get_safe_default_features(numeric_cols)
    assert defaults == ['Feature_A', 'Feature_B']
    assert 'Annual Income (k$)' not in defaults

def test_prepare_feature_matrix_clean():
    df = pd.DataFrame({
        'X': [1.0, 2.0, 3.0, np.nan, 5.0],
        'Y': [10.0, 20.0, 30.0, 40.0, np.inf]
    })
    X_clean, valid_mask, err = prepare_feature_matrix(df, ['X', 'Y'])
    assert err is None
    assert len(X_clean) == 3
    assert valid_mask.sum() == 3

def test_calculate_elbow_normal():
    df = pd.read_csv('Mall_Customers.csv')
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    res = calculate_elbow(X, max_k=10, random_state=0)
    assert len(res['k_range']) == 10
    assert len(res['wcss']) == 10
    assert res['wcss'][0] > res['wcss'][4]  # WCSS monotonically decreases
    assert res['optimal_k_by_silhouette'] == 5

def test_calculate_elbow_small_dataset():
    small_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    res = calculate_elbow(small_df, max_k=10, random_state=0)
    assert len(res['k_range']) <= 3  # samples - 1
    assert max(res['k_range']) < 10

def test_fit_kmeans_model():
    df = pd.read_csv('Mall_Customers.csv')
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    model, labels, sil, inertia = fit_kmeans_model(X, n_clusters=5, random_state=0)
    assert len(labels) == 200
    assert set(labels) == {0, 1, 2, 3, 4}
    assert model.cluster_centers_.shape == (5, 2)
    assert sil > 0.5
    assert inertia < 45000

def test_generate_cluster_insights():
    df = pd.read_csv('Mall_Customers.csv')
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    model, labels, _, _ = fit_kmeans_model(X, n_clusters=5, random_state=0)
    summary_df, insights = generate_cluster_insights(df, features, labels, k=5)
    
    assert len(summary_df) == 5
    assert len(insights) == 5
    assert summary_df['Customer Count'].sum() == 200
    
    # Check that cluster 0 is labeled Moderate / Moderate
    c0 = summary_df[summary_df['Cluster ID'] == 0].iloc[0]
    assert c0['Annual Income (k$) Tier'] == 'Moderate'
    assert c0['Spending Score (1-100) Tier'] == 'Moderate'

def test_cluster_averages_match_underlying_data():
    df = pd.read_csv('Mall_Customers.csv')
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    _, labels, _, _ = fit_kmeans_model(X, n_clusters=5, random_state=0)
    summary_df, _ = generate_cluster_insights(df, features, labels, k=5)

    # Check reported averages exactly equal mathematical mean of slice
    for c_id in range(5):
        c_mask = (labels == c_id)
        expected_f1 = round(df.loc[c_mask, features[0]].mean(), 2)
        expected_f2 = round(df.loc[c_mask, features[1]].mean(), 2)
        
        row = summary_df[summary_df['Cluster ID'] == c_id].iloc[0]
        assert row[f"Avg {features[0]}"] == expected_f1
        assert row[f"Avg {features[1]}"] == expected_f2
        assert row["Customer Count"] == c_mask.sum()

def test_custom_dataset_arbitrary_columns():
    csv_text = "Metric_A,Metric_B,Category\n10,100,Alpha\n20,200,Beta\n30,300,Alpha\n40,400,Beta\n50,500,Alpha\n"
    df, err = load_dataset(io.StringIO(csv_text))
    assert err is None
    num_cols = get_numeric_columns(df)
    assert num_cols == ['Metric_A', 'Metric_B']
    
    defaults = get_safe_default_features(num_cols)
    assert defaults == ['Metric_A', 'Metric_B']
    
    X_clean, valid_mask, prep_err = prepare_feature_matrix(df, defaults)
    assert prep_err is None
    assert len(X_clean) == 5
    
    model, labels, sil, inertia = fit_kmeans_model(X_clean, n_clusters=2, random_state=0)
    assert len(labels) == 5
    assert set(labels) == {0, 1}

def test_fit_kmeans_with_varying_cluster_counts():
    df = pd.read_csv('Mall_Customers.csv')
    X = df[['Age', 'Spending Score (1-100)']]
    
    for k in [2, 3, 4, 6]:
        model, labels, sil, inertia = fit_kmeans_model(X, n_clusters=k, random_state=42)
        assert len(labels) == 200
        assert len(set(labels)) == k
        assert model.cluster_centers_.shape == (k, 2)
        assert inertia > 0
        assert -1.0 <= sil <= 1.0
