# 🛍️ Customer Segmentation using K-Means (Streamlit App)

This project performs **customer segmentation** using **K-Means Clustering**, and builds an interactive **Streamlit web app** for visual analysis and business insights.

It helps businesses understand different types of customers like:
- 🟥 High-income, high-spending customers (VIP / Target Group)
- 🟦 Low-income, low-spending customers (Budget-conscious)
- 🟪 Impulsive buyers (Low-income, high-spending)
- 🟩 Cautious but wealthy customers (High-income, low-spending)
- 🟧 Balanced / Mainstream customers (Moderate-income, moderate-spending)

---

## 📌 Objective

🎯 To segment customers into useful clusters using machine learning and visualize results dynamically.  
The tool allows:
- Marketing teams to design **targeted strategies**
- Businesses to **personalize offers**
- Analysts to **explore patterns** in customer behavior

## LIVE : https://customer-segmentation-using-kmeans-clustering.streamlit.app/

---

## 🚀 Features

✅ Upload your own dataset or use the bundled `Mall_Customers.csv`  
✅ Select any 2 numeric features (Age, Income, Spending Score, etc.) with safe auto-detection  
✅ Elbow Method & Silhouette analysis to choose the optimal number of clusters (K)  
✅ K-Means clustering with deterministic seeds for reproducible results  
✅ Dynamic scatter plot with distinct cluster colors and visible **Centroids (⭐)**  
✅ Real-time KPI metric cards (Total Customers, Clusters K, Silhouette Score, WCSS)  
✅ Smart report with cluster-wise statistical distributions and customer shares  
✅ Styled table showing cluster means and distributions  
✅ Business-friendly persona archetypes and targeted marketing recommendations  
✅ Download clustered results as a CSV file  

---

## 🧪 Technologies Used

| Tool / Library     | Purpose                             |
|--------------------|--------------------------------------|
| Python             | Main programming language            |
| Pandas             | Data analysis & cleaning             |
| Matplotlib         | Elbow graph & cluster visualization  |
| Seaborn            | Enhanced styling & color palettes    |
| Scikit-learn       | K-Means clustering & Silhouette Score|
| Streamlit          | UI for interactive web app           |
| Pytest             | Automated unit testing               |

---

## 📊 Dataset Overview

We used `Mall_Customers.csv`, containing 200 records:

| Column                | Description                         |
|------------------------|-------------------------------------|
| CustomerID            | Unique customer ID (Index)          |
| Gender                | Male / Female                       |
| Age                   | Age of customer (18 – 70)           |
| Annual Income (k$)    | Income in thousands of dollars      |
| Spending Score (1–100)| Score assigned by the shopping mall |

---

## 💻 How to Run Locally

### 1. Activate Environment & Install Dependencies
```bash
# Using the project's virtual environment
.\.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Jupyter Notebook
```bash
jupyter notebook main.ipynb
# or run in VS Code / Cursor by selecting the .venv kernel
```

### 4. Run Automated Unit Tests
```bash
pytest -v tests/test_clustering_helper.py
```

---

## 🖼️ How It Works

1. **Select Dataset Source**: Use the built-in `Mall_Customers.csv` or upload a custom CSV.
2. **Feature Selection**: Pick any 2 continuous numerical features for clustering.
3. **Elbow Method Analysis**: Examine the WCSS curve and peak Silhouette Score to pick the best number of clusters (K).
4. **Cluster Visualization**: View customers in color-coded clusters with yellow star centroids (⭐).
5. **Cluster Personas & Report**: Read data-grounded persona archetypes and tailored marketing actions.
6. **Download Data**: Export the segmented customer list with assigned cluster labels.

---

## 🎓 Viva Quick Reference

- **K-Means Algorithm**: An unsupervised centroid-based partition clustering algorithm that groups $N$ observations into $K$ clusters by minimizing the within-cluster sum of squares (WCSS / Inertia).
- **Elbow Method**: A heuristic where WCSS is plotted against $K$. The bend ("elbow") indicates where diminishing returns set in for adding further clusters.
- **Silhouette Score**: Ranges from -1 to +1. Measures how similar an object is to its own cluster (cohesion) compared to neighboring clusters (separation). A score of **0.5539** at $K=5$ indicates strong cluster quality.
