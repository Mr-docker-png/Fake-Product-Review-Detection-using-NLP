# Fake Product Review Detection using NLP

A Natural Language Processing (NLP) and Machine Learning project that classifies product reviews as **Original/Genuine (`OR`)** or **Computer-Generated/Fake (`CG`)**.

The project uses **TF-IDF vectorization** to convert review text into numerical features and compares multiple classical Machine Learning algorithms before selecting Linear Support Vector Machine (SVM) as the final model.

---

## Project Overview

Fake and computer-generated product reviews can make it difficult for customers to distinguish genuine opinions from artificially generated content.

This project builds a Machine Learning classification system that analyzes the text of a product review and predicts whether it belongs to:

- `OR` — Original/Genuine Review
- `CG` — Computer-Generated/Fake Review

The complete workflow is:

```text
Product Review Dataset
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Null Value Checking
        ↓
Train/Test Split
        ↓
TF-IDF Vectorization
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
5-Fold Cross-Validation
        ↓
GridSearchCV
        ↓
Final Linear SVM
        ↓
Save Model
        ↓
Predict New Reviews
Dataset

The project uses a Fake Reviews dataset containing approximately 40,420 reviews after duplicate removal.

Dataset Classes
Label	Meaning	Count
OR	Original/Genuine Review	20,215
CG	Computer-Generated/Fake Review	20,205

The dataset is therefore approximately balanced between the two classes.

Dataset Columns
Column	Description
category	Product/review category
rating	Review rating
label	Target class (OR or CG)
text_	Review text

For the main NLP experiment, the text_ column is used as the input feature and label is used as the target.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Natural Language Processing
TF-IDF
Jupyter Notebook
Data Preparation

The dataset was checked before model training.

The following preprocessing steps were performed:

Removed duplicate rows
Checked for missing/null values
Inspected review text
Examined review length
Separated text features and target labels
Used stratified train/test splitting

After cleaning:

Total reviews: 40,420
Train/Test Split

The dataset was divided into:

80% Training
20% Testing

Using:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Final split:

Training samples: 32,336
Testing samples:   8,084

stratify=y was used to maintain the class distribution between the training and testing sets.

Text Feature Extraction

Machine Learning algorithms cannot directly process raw review text.

Therefore, TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the reviews into numerical feature vectors.

Review Text
     ↓
TF-IDF Vectorizer
     ↓
Numerical Sparse Matrix
     ↓
Machine Learning Model

The TF-IDF vectorizer generated:

36,824 features

Resulting dimensions:

Training: (32336, 36824)
Testing:  (8084, 36824)

The vectorizer was fitted only on the training data:

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

This prevents information from the test set from being used during feature learning.

Machine Learning Models

Three Machine Learning algorithms were evaluated.

1. Multinomial Naive Bayes

Multinomial Naive Bayes was used as the baseline text-classification model.

Accuracy
87.00%
Classification Report
              precision    recall  f1-score   support

CG               0.82      0.93      0.87      4041
OR               0.92      0.80      0.86      4043

accuracy                              0.87      8084
macro avg          0.87      0.87      0.87      8084
weighted avg       0.87      0.87      0.87      8084
Confusion Matrix
[[3773  268]
 [ 815 3228]]
2. Logistic Regression

Logistic Regression was then trained using the TF-IDF features.

Accuracy
90.57%
Classification Report
              precision    recall  f1-score   support

CG               0.91      0.89      0.90      4041
OR               0.90      0.92      0.91      4043

accuracy                              0.91      8084
macro avg          0.91      0.91      0.91      8084
weighted avg       0.91      0.91      0.91      8084
Confusion Matrix
[[3615  426]
 [ 336 3707]]
3. Linear Support Vector Machine

A Linear SVM was trained because linear classifiers are well suited to high-dimensional sparse TF-IDF text representations.

Accuracy
91.09%
Classification Report
              precision    recall  f1-score   support

CG               0.91      0.91      0.91      4041
OR               0.91      0.91      0.91      4043

accuracy                              0.91      8084
macro avg          0.91      0.91      0.91      8084
weighted avg       0.91      0.91      0.91      8084
Confusion Matrix
[[3693  348]
 [ 372 3671]]
Model Comparison
Model	Test Accuracy
Multinomial Naive Bayes	87.00%
Logistic Regression	90.57%
Linear SVM	91.09%

The Linear SVM produced the highest test accuracy among the three models tested.

Cross-Validation

5-fold cross-validation was performed on the Linear SVM.

Fold Scores
Fold 1: 90.88%
Fold 2: 90.85%
Fold 3: 90.34%
Fold 4: 90.21%
Fold 5: 90.30%
Mean CV Accuracy
90.52%
Standard Deviation
0.286%

The relatively small standard deviation indicates that the validation scores were consistent across the five folds.

Hyperparameter Tuning

GridSearchCV was used to tune the C parameter of LinearSVC.

Parameters tested:

C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
Best Parameter
C = 1
Best Cross-Validation Accuracy
90.52%

The best value was C=1, which is also the default value used by LinearSVC().

Therefore, hyperparameter tuning did not improve the model beyond the original Linear SVM configuration.

Final Model

The final model is:

TF-IDF Vectorizer
        +
LinearSVC(C=1)
Final Test Performance
Accuracy: 91.09%
Final Confusion Matrix
[[3693  348]
 [ 372 3671]]

This corresponds to:

3,693 CG reviews correctly classified
3,671 OR reviews correctly classified
348 CG reviews classified as OR
372 OR reviews classified as CG
Saving the Model

The trained TF-IDF vectorizer and final Linear SVM model were saved using Joblib.

tfidf_vectorizer.pkl
fake_review_model.pkl

These files allow the model to be loaded later without retraining.

Example:

import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("fake_review_model.pkl")
Making Predictions

New reviews can be transformed using the saved TF-IDF vectorizer and passed to the trained model.

Example:

reviews = [
    "Absolutely amazing product! The quality is excellent and I am very happy with my purchase.",
    "This product is terrible. It stopped working after only two days and the quality is very poor."
]

reviews_tfidf = vectorizer.transform(reviews)

predictions = model.predict(reviews_tfidf)

for review, prediction in zip(reviews, predictions):
    print("Review:", review)
    print("Prediction:", prediction)

The model returns one of:

OR

or

CG

where:

OR = Original/Genuine
CG = Computer-Generated/Fake

Clone the repository:

git clone https://github.com/YOUR-USERNAME/Fake-Product-Review-Detection.git

Navigate to the project:

cd Fake-Product-Review-Detection

Install the dependencies:

pip install -r requirements.txt

Run Jupyter Notebook:

jupyter notebook

Open:

main.ipynb
Requirements

The main dependencies are:

pandas
numpy
scikit-learn
joblib
jupyter

See requirements.txt for the complete list.

Key Concepts Learned

This project demonstrates practical understanding of:

Natural Language Processing
Text classification
TF-IDF
Sparse matrices
Train/Test Split
Stratified splitting
Multinomial Naive Bayes
Logistic Regression
Linear SVM
Confusion Matrix
Precision
Recall
F1-score
Cross-validation
GridSearchCV
Hyperparameter tuning
Model persistence with Joblib
Prediction on unseen text
Limitations

The model is based primarily on textual patterns learned from the training dataset.

Therefore:

A positive review is not automatically genuine.
A negative review is not automatically fake.
The prediction reflects patterns learned from the dataset.
Individual predictions should not be treated as definitive proof that a review is fake or genuine.
Performance on reviews from a different source or domain may differ from the reported test performance.
Future Improvements

Possible extensions include:

Build a Streamlit web application
Add character-level TF-IDF
Experiment with word n-grams
Experiment with character n-grams
Perform additional text preprocessing
Compare additional NLP models
Add probability/confidence visualization where supported
Build a FastAPI prediction API
Deploy the model online
Add an interactive review-analysis dashboard
Conclusion

This project implements a complete classical NLP pipeline for detecting computer-generated product reviews.

Three Machine Learning models were compared:

Multinomial Naive Bayes → 87.00%
Logistic Regression     → 90.57%
Linear SVM              → 91.09%

The Linear SVM achieved 91.09% accuracy on the held-out test set.

5-fold cross-validation produced a mean accuracy of 90.52%, and GridSearchCV selected C=1 as the best tested parameter.

The final trained components were saved as:

tfidf_vectorizer.pkl
fake_review_model.pkl
Author: Jaskaran Singh
Machine Learning / Python Developer
License
This project is intended for educational and portfolio purposes.
