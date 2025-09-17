# Water Quality Prediction Model

[![GitHub stars](https://img.shields.io/github/stars/yourusername/water-quality-prediction?style=social)](https://github.com/abubakarpungiwale/water-quality-prediction/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/water-quality-prediction?style=social)](https://github.com/abubakarpungiwale/water-quality-prediction/network)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This repository presents a machine learning solution for predicting water potability using a dataset of 3276 samples and 10 features (e.g., pH, Hardness, Chloramines). Developed during a Data Scientist internship at **Milestone PLM Solutions Pvt. Ltd., Thane**, it demonstrates an end-to-end pipeline for binary classification (0: unsafe, 1: safe), including EDA, preprocessing, model training, and evaluation.

**Key Highlights**:
- **Ensemble Modeling**: Random Forest as the top-performing ensemble for robust predictions.
- **Multiple Algorithms**: Comparative analysis of Logistic Regression, KNN, Naive Bayes, Random Forest, and SVM.
- **Scalability**: Potential for neural networks or advanced ensembles in extensions.

This project showcases classification expertise, feature scaling, and model optimization, ideal for data science and ML engineering roles.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Technologies](#key-technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Key Technologies

- **Libraries**: Pandas, Seaborn, Matplotlib, Scikit-learn (for preprocessing, models, and evaluation), Joblib (for model saving).
- **Models**: Logistic Regression, K-Nearest Neighbors (k=5), Gaussian Naive Bayes, Random Forest (n_estimators=100, ensemble), SVM.
- **Techniques**: StandardScaler for normalization, train_test_split (70/30), mean imputation grouped by target.
- **Metrics**: Accuracy, Confusion Matrix, Classification Report.
- **Visualization**: Boxplots, correlation heatmaps.

## Installation

```bash
git clone https://github.com/yourusername/water-quality-prediction.git
cd water-quality-prediction
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- Packages: pandas, seaborn, matplotlib, scikit-learn, joblib.

## Usage

1. Place `water_potability.csv` in the root directory.
2. Run:
   ```bash
   jupyter notebook Water_Quality_Prediction.ipynb
   ```
3. Execute cells for EDA, model training, and evaluation. Saved model: `svm.pkl` (extendable to RF).

## Methodology

- **Preprocessing**: EDA (head, shape, null checks), mean imputation by Potability, StandardScaler.
- **Feature Analysis**: Boxplots for feature distributions by class, correlation heatmap.
- **Training**: Split data (70% train, 30% test), trained 5 classifiers.
- **Ensemble**: Random Forest as bagging ensemble for improved accuracy and reduced overfitting.

## Performance Metrics

- **Model Comparison** (Accuracy on Test Set):
  - Random Forest: ~67% (highest, ensemble strength).
  - SVM: ~66%.
  - Logistic Regression: ~64%.
  - Others: 60-65%.
- **Insights**: Confusion matrices and reports show balanced precision/recall. Visualizations in notebook enhance interpretability.

## Contributing

Fork and submit pull requests for enhancements like neural networks.

## License

MIT License - see [LICENSE](LICENSE).

## Contact

- **Author**: Abubakar Maulani Pungiwale
- **Email**: your.email@example.com
- **LinkedIn**: [linkedin.com/in/abubakarpungiwale](https://linkedin.com/in/abubakarpungiwale)
- **Contact**: +91 9321782858
Connect for ML discussions or data science opportunities!

---
