## Project: Identify Customer Segments (Otto Group / Arvato)

This project uses unsupervised learning to analyze demographic data from a German mail-order company, provided by Bertelsmann Arvato Analytics. The goal is to identify population segments that are most likely to become customers, which can then inform a mail-out marketing campaign.

### 📈 Methodology

1.  **Data Loading:**
    * `Udacity_AZDIAS_Subset.csv`: Demographic data for the general population of Germany.
    * `Udacity_CUSTOMERS_Subset.csv`: Demographic data for the mail-order company's customers.
    * `AZDIAS_Feature_Summary.csv`: A data dictionary used to identify missing or unknown value codes.

2.  **Data Preprocessing (Data Cleaning Function):**
    * Missing values identified in the feature summary were converted to `NaN`.
    * Columns with a high percentage of missing data (e.g., `AGER_TYP`, `TITEL_KZ`) were dropped.
    * Rows with a high number of missing values were assessed and separated.

3.  **Feature Engineering & Re-Encoding:**
    * **Categorical Features:** Binary categorical features (e.g., `ANREDE_KZ`, `OST_WEST_KZ`) were re-encoded, and multi-level features were one-hot encoded.
    * **Mixed-Type Features:** Complex features were engineered into new, usable variables:
        * `PRAEGENDE_JUGENDJAHRE`: Split into 'decade' and 'movement' (mainstream vs. avantgarde).
        * `CAMEO_INTL_2015`: Split into 'wealth' and 'life_stage' based on its digits.

4.  **Modeling (Unsupervised Learning):**
    * **Feature Scaling:** A `StandardScaler` was fit to the cleaned general population data to normalize features.
    * **Dimensionality Reduction:** **PCA (Principal Component Analysis)** was applied to reduce the number of features while retaining a high percentage of variance.
    * **Clustering:** **K-Means clustering** was used on the PCA-transformed data to group the general population into distinct segments. The optimal number of clusters was determined using the "Elbow Method" by plotting the sum of squared errors (SSE).

### 🎯 Outcome

By comparing the cluster proportions of the general population to the customer population, this analysis identifies which segments are **over-represented** (the target audience) and **under-represented** (less likely to be customers).

---

### 🛠️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    ```
2.  **Install dependencies:**
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `scikit-learn`
    * `torch`
    * `torchvision`
    * `jupyter notebook`

3.  **Run the notebooks:**
    * Navigate to the project directory and launch Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    * Open `Identify_Customer_Segments.ipynb` or `CIFAR-10_Image_Classifier.ipynb`.
    * **Note for Project 1:** You will need the `.csv` data files (`Udacity_AZDIAS_Subset.csv`, `Udacity_CUSTOMERS_Subset.csv`, and `AZDIAS_Feature_Summary.csv`) in the same directory to run the notebook.
