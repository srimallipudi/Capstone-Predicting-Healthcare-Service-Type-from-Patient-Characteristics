# Predictive Healthcare Service Utilization

### Project Overview:
This project focuses on predicting the type of healthcare service a patient might require whether it be Inpatient / Outpatient / Emergency / Residential care based on their demographics, socio-economic factors, and medical conditions. By analyzing a comprehensive dataset comprising patient records, including demographic information, medical history, and healthcare service utilization patterns, we aim to develop predictive models that can assist healthcare providers in efficiently allocating resources and enhance timely & appropriate care for patients.

### Data Exploration:
We begin by exploring the Patient Characteristics Survey dataset, which includes information on 196,102 patients. Outpatient care emerges as the most utilized service, with approximately 68% of patients preferring this option. We observe demographic trends, regional disparities, and ethnic influences on healthcare utilization. Insurance type significantly impacts healthcare choices, and certain medical conditions are more prevalent among specific patient groups.

### Data Pre-Processing:
The dataset undergoes several pre-processing steps to ensure data quality and compatibility for modeling. We integrate additional datasets by zip code to incorporate other relevant social and environmental factors such as Median house income, number of hospitals, number of parks in that particular zip code. Irrelevant columns are removed, and missing values are handled using various strategies such as imputation and dropping rows. We perform normalization and feature selection techniques to prepare the dataset for modeling.

### Modeling and Evaluation:
We train various machine learning models, including Random Forest, Decision Tree, Gradient Boosting, Neural Networks, Logistic Regression, and Naive Bayesian, on three distinct datasets: Original, Oversampled, and Oversampled with PCA. Models are evaluated based on accuracy, precision, recall, and F1 score. Random Forest, Decision Tree, and Gradient Boosting demonstrate strong predictive capabilities for healthcare service utilization.

#### Table:
| Classification Model | Precision | Recall | Accuracy | Computation Time (in sec) |
|----------------------|-----------|--------|----------|---------------------------|
| Random Forest        |   0.98    |  0.97  |   0.97   |          215.25           |
| Decision Tree        |   0.96    |  0.96  |   0.96   |           22.84           |
| Gradient Boosting    |   0.78    |  0.79  |   0.79   |         2322.39           |

### Methodological Contributions:
Throughout the project, we implement rigorous methodologies to ensure reliable results. We employ Train-Validate-Test splits and stratified sampling for model evaluation. Additionally, we experiment with oversampling techniques and Principal Component Analysis (PCA) for dimensionality reduction to enhance model performance. We also explored k-means clustering for pattern identification in the dataset.

### Conclusions and Recommendations:
Our findings suggest that predictive models can effectively forecast healthcare service utilization based on patient characteristics. However, challenges remain in accurately predicting emergency services and incorporating temporal changes in patient demographics and healthcare trends. We recommend updating the dataset to capture the latest trends, conducting external validation for broader applicability, and collaborating with healthcare institutions across diverse regions to enhance model robustness.

### Future Projects:
Future research endeavors could focus on developing specialized models for different care needs to further refine predictive capabilities in diverse healthcare settings. Collaboration with healthcare institutions across diverse regions to collect region-specific data would also contribute to enhancing the model's generalizability and predictive accuracy. Additionally, incorporating real-time data streams and advanced machine learning techniques could further improve the accuracy and reliability of predictive models for healthcare service utilization.

####  Attachments:

[Capstone Project Paper.pdf](https://github.com/srimallipudi/Capstone-Project-Predicting-Healthcare-Service-Type-from-Patient-Characteristics-and-Medical-History/files/14782784/Capstone.Project.Paper.pdf)

[Capstone Project - Presentation.pdf](https://github.com/srimallipudi/Capstone-Project-Predicting-Healthcare-Service-Type-from-Patient-Characteristics-and-Medical-History/files/14782786/Capstone.Project.-.Presentation.pdf)
