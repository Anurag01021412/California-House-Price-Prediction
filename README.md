# California House Price Prediction
This project predicts median house values in California using the California Housing Dataset. The workflow includes data preprocessing, feature engineering, and model training using a RandomForestRegressor to achieve robust and accurate predictions.
# Features
**Data Preprocessing Pipeline**
Missing value imputation (median strategy)
Feature scaling for numerical variables
One-hot encoding for categorical variables
**Stratified Sampling**
Stratified split based on income categories to ensure balanced training and test sets.
**Modeling**
Random Forest Regression with hyperparameter tuning
Cross-validation to evaluate model performance
**Performance Metric**
Achieved cross-validated RMSE of ~$49K
**Automation**
Full pipeline and trained model saved with joblib for reproducibility
Supports inference on new datasets (input.csv → output.csv)


# Note: The trained model file (model.pkl) exceeds GitHub’s 100 MB file size limit and could not be uploaded directly to the repository.
# 📂 Full Project (including dataset and artifacts): https://drive.google.com/drive/folders/1eNUi0xZJRgV_xTdLA6LjBcbP9Gmok5Xu?usp=share_link
# 💾 Trained Model (model.pkl): https://drive.google.com/file/d/1_EmFVBvK3KA8p-ULSfV3IX2m7psqqbFQ/view?usp=share_link
