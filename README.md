# Credit Risk Assessment Model

## Overview
This project involves developing a predictive model to assess loan default risks. Using a dataset that includes various borrower characteristics, the model predicts whether a loan will default based on historical data.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NumPy
- Matplotlib/Seaborn (for visualizations)
- Jupyter Notebook (for development)

## Dataset
The dataset used for this project contains information about borrowers, including features like age, income, employment length, loan amount, interest rate, and loan status. It consists of **32,581** entries with various attributes relevant to credit risk.

## Features
- `person_age`: Age of the borrower
- `person_income`: Income of the borrower
- `person_emp_length`: Employment length (in years)
- `loan_amnt`: Amount of the loan
- `loan_int_rate`: Interest rate of the loan
- `loan_status`: Status of the loan (0 for non-default, 1 for default)
- Additional features related to borrower demographics and credit history

## Modeling Approach
1. **Data Preprocessing**: Clean and preprocess the data, handle missing values, and encode categorical variables.
2. **Train-Test Split**: Divide the dataset into training and testing sets.
3. **Model Selection**: Use logistic regression to model the likelihood of default.
4. **Hyperparameter Tuning**: Optimize the model using techniques such as grid search.
5. **Evaluation**: Evaluate the model's performance using accuracy, confusion matrix, and classification report.

## Results
- **Accuracy**: 86.49%
- **Confusion Matrix**:
  
|                  | Predicted: No (0) | Predicted: Yes (1) |
|------------------|-------------------|---------------------|
| Actual: No (0)   | True Negatives     | False Positives      |
| Actual: Yes (1)  | False Negatives    | True Positives       |
- **Classification Report**: Detailed performance metrics (precision, recall, F1-score).

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/lakshyajoshii/credit_risk_model.git
2. Navigate to the project directory:
    ```bash
    cd credit_risk_model
3. Run the main script:
    ```bash
    python main.py
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
    
