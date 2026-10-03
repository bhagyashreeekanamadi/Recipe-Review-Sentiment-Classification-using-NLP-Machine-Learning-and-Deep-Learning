# Recipe Review Sentiment Classification using NLP, Machine Learning and Deep Learning

## Project Overview

This project focuses on classifying recipe reviews into **Positive** and **Negative** sentiment categories using Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques.

The project uses a Recipe Reviews and User Feedback dataset containing user reviews, recipe information, ratings, and feedback-related features.

The primary objective is to process review text, build multiple sentiment classification models, compare their performance, and identify the model that performs best on validation data.

It also includes error analysis, inference time comparison, and a function to predict the sentiment of new recipe reviews.

## Objectives

* Perform text preprocessing and exploratory data analysis.
* Convert review text into numerical representations using TF-IDF and tokenization.
* Build a traditional Machine Learning sentiment classifier.
* Implement ANN, CNN, and RNN models for text classification.
* Compare models using multiple evaluation metrics.
* Analyze incorrectly classified reviews.
* Compare model inference times.
* Predict sentiment for new user-provided reviews.

## Dataset

**Dataset:** Recipe Reviews and User Feedback Dataset

The dataset contains approximately 18,182 records and 15 columns.

Important features include:

| Column            | Description                   |
| ----------------- | ----------------------------- |
| `recipe_number`   | Recipe identifier             |
| `recipe_code`     | Recipe code                   |
| `recipe_name`     | Name of the recipe            |
| `user_id`         | User identifier               |
| `user_reputation` | User reputation               |
| `reply_count`     | Number of replies             |
| `thumbs_up`       | Positive feedback on a review |
| `thumbs_down`     | Negative feedback on a review |
| `stars`           | Rating given to the recipe    |
| `text`            | User-written review           |

### Target Variable: Sentiment

The `stars` column is used to create sentiment labels.

| Star Rating | Sentiment | Encoded Label |
| ----------- | --------- | ------------: |
| 1–2         | Negative  |             0 |
| 4–5         | Positive  |             1 |
| 0 and 3     | Excluded  |             - |

The review text (`text`) is used as the input feature, while the rating-derived sentiment is the target.

**Important dataset limitation:** These labels are derived from star ratings rather than manually annotated review text. Therefore, the project predicts rating-derived sentiment. A review's wording may not always agree with its star rating.

The dataset also has class imbalance, with substantially more positive reviews than negative reviews. Class weighting is used during model training to help address this imbalance.

## Technologies and Libraries

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* NLTK-style text preprocessing concepts
* Joblib

## Project Workflow

The project follows this workflow:

1. Dataset loading and inspection
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Sentiment label creation
5. Train, validation, and test splitting
6. Text vectorization and tokenization
7. Machine Learning model development
8. Deep Learning model development
9. Model evaluation and comparison
10. Confusion matrix analysis
11. Error analysis
12. Inference time comparison
13. Model saving
14. Prediction on new reviews

## Models Implemented

### 1. Logistic Regression

A traditional Machine Learning classification algorithm.

**Text representation:** TF-IDF

**Purpose:** Establish a Machine Learning baseline for sentiment classification.

### 2. Artificial Neural Network (ANN)

A feed-forward neural network for text classification.

**Architecture:**

* Embedding layer
* Global Average Pooling
* Dense layers
* Dropout
* Sigmoid output layer

### 3. Convolutional Neural Network (CNN)

A one-dimensional CNN that learns local patterns and feature combinations in token sequences.

**Architecture:**

* Embedding layer
* Conv1D
* Global Max Pooling
* Dense layers
* Dropout
* Sigmoid output layer

### 4. Recurrent Neural Network (RNN)

A SimpleRNN-based model that processes token sequences while maintaining information from preceding sequence positions.

**Architecture:**

* Embedding layer
* SimpleRNN
* Dropout
* Dense layers
* Sigmoid output layer

## Model Evaluation

The models are evaluated using a validation set and a separate test set.

The following metrics are calculated:

| Metric      | Purpose                                                     |
| ----------- | ----------------------------------------------------------- |
| Accuracy    | Measures overall correct predictions                        |
| Precision   | Measures the correctness of positive predictions            |
| Recall      | Measures how many actual positive cases are identified      |
| F1 Score    | Balances precision and recall                               |
| Negative F1 | Evaluates negative sentiment classification                 |
| Positive F1 | Evaluates positive sentiment classification                 |
| Macro F1    | Gives equal importance to both classes                      |
| ROC-AUC     | Measures the ability to distinguish between the two classes |

**Model selection:** Macro F1 on the validation set is used to select the best model.

**Final evaluation:** The selected and other trained models are evaluated on the held-out test set.

Macro F1 is particularly important because the dataset contains significantly more positive reviews than negative reviews.

## Error Analysis

Error analysis is performed using the selected model.

It identifies incorrectly classified reviews by comparing actual sentiment labels with predicted sentiment labels.

The analysis includes:

* Original review text
* Actual sentiment
* Predicted sentiment
* Predicted positive probability

This helps identify cases where the model struggles to distinguish between positive and negative reviews.

## Inference Time Comparison

The project measures the time required by each model to generate predictions for the test dataset.

It records:

* Total batch prediction time
* Approximate average prediction time per review

These measurements provide a basic comparison of model prediction speed. They are benchmark measurements from the notebook environment, not guaranteed real-world production latency.

## Predict Sentiment on New Reviews

A prediction function is implemented to classify newly entered recipe reviews.

Example:

```python
review = "This recipe is delicious. My family loved it!"

result = predict_sentiment(review)

print(result)
```

Example output:

```python
{
    'review': 'This recipe is delicious. My family loved it!',
    'predicted_sentiment': 'Positive',
    'positive_probability': 0.98
}
```

*The probability shown is illustrative; actual output depends on the trained model.*

## Model and Results Saving

The project saves trained models and evaluation outputs.

### Models

* `logistic_regression.pkl`
* `tfidf_vectorizer.pkl`
* `ann_model.keras`
* `cnn_model.keras`
* `rnn_model.keras`
* `best_model.keras` or the corresponding best Logistic Regression artifacts
* `tokenizer.json`

### Results

* `validation_comparison.csv`
* `model_comparison.csv`
* `error_analysis.csv`
* `inference_time.csv`
* Confusion matrix images

## How to Run the Project

### Option 1: Google Colab

1. Open Google Colab.
2. Upload the project notebook.
3. Upload the Recipe Reviews and User Feedback CSV dataset.
4. Run the notebook cells sequentially.
5. Review the EDA visualizations and model training results.
6. Compare the evaluation metrics.
7. Test the prediction function with new reviews.

### Option 2: Local Environment

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

Open the notebook using Jupyter Notebook or JupyterLab and provide the dataset path before executing the cells.

## Suggested Repository Structure

```text
recipe-review-sentiment-analysis/
│
├── dataset/
│   └── Recipe Reviews and User Feedback Dataset.csv
│
├── notebooks/
│   └── recipe_review_sentiment_analysis.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── ann_model.keras
│   ├── cnn_model.keras
│   ├── rnn_model.keras
│   └── tokenizer.json
│
├── results/
│   ├── validation_comparison.csv
│   ├── model_comparison.csv
│   ├── error_analysis.csv
│   ├── inference_time.csv
│   └── confusion_matrix_*.png
│
├── README.md
└── requirements.txt
```

*The folder structure represents the suggested organization of the project. Include generated model and result files after running the notebook.*

## Key Learning Outcomes

Through this project, I explored:

* Text preprocessing and NLP fundamentals.
* TF-IDF feature extraction.
* Tokenization and sequence padding.
* Traditional Machine Learning for text classification.
* ANN, CNN, and RNN architectures.
* Handling class imbalance using class weights.
* Model evaluation using classification metrics.
* Confusion matrix interpretation.
* Error analysis and misclassification inspection.
* Model comparison and inference time measurement.
* Saving trained models and performing predictions on new data.

## Future Improvements

* Experiment with different classification thresholds.
* Perform hyperparameter tuning.
* Explore pretrained word embeddings.
* Experiment with LSTM and GRU architectures.
* Investigate transformer-based sentiment classification.
* Use manually annotated sentiment labels for more reliable evaluation.
* Explore explainability techniques to understand model predictions.

## Conclusion

This project demonstrates an end-to-end sentiment classification workflow using NLP, Machine Learning, and Deep Learning.

By implementing Logistic Regression, ANN, CNN, and RNN, the project compares different approaches to text classification and examines their performance using metrics that account for class imbalance.

The project also includes error analysis, inference benchmarking, and prediction on new recipe reviews.

---

**Author:** Bhagyashree Kanamadi

**Focus Areas:** NLP | Machine Learning | Deep Learning | Text Classification | Model Evaluation
