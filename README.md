# Relation Extraction on TACRED Dataset

## Project Overview

This project evaluates different approaches for relation extraction (RE) on the TACRED dataset. The implementation compares:

1. **Deep learning without transformers**: Long Short-Term Memory (LSTM)
2. **Deep learning with transformers**: BERT
3. **Traditional machine learning**: Support Vector Machine (SVM)

Relation extraction identifies and categorizes semantic relationships between named entities in text. The TACRED dataset contains 106,264 instances across 42 relation types, with approximately 80% labeled as 'no_relation'. The models were evaluated using precision, recall, and F1-score metrics.

## Key Findings

- **Two-stage Classification**: Splitting the task into relation detection and relation classification significantly improved performance for all models.
- **Model Performance**: 
  - LSTM achieved the highest macro F1 score (0.61) on relation-only data, slightly outperforming BERT (0.57)
  - BERT performed best on the full dataset with a weighted F1 score of 0.81
- **Sequence Length**: Model performance was generally consistent across different sentence lengths, with only BERT showing a decrease in performance for longer sequences.
- **Class Imbalance**: Severe class imbalance (80% no_relation) remains a significant challenge for all models.

## Folder Structure

```
|— user_interface.ipynb        # This notebook serves as the user interface
|— choose_model_test.py        # This script loads the pre-trained models for prediction
|— data                        # This folder contains the dataset
    |— train.json
    |— test.json
    |— dev.json
|— models                      # This folder contains the model implementations
    |— bert_relation_only.py   # BERT model for relation-only classification
    |— bert.py                 # BERT model for all relations classification
    |— lstm_relation_only.py   # LSTM model for relation-only classification
    |— lstm.py                 # LSTM model for all relations classification
    |— svm_predict.py          # SVM model for all relations classification
|— models_parameters           # This folder contains the weights for pre-trained models
    |— bertmodel.pt            # Pre-trained BERT weights for all relations
    |— bertmodelnorelation.pt  # Pre-trained BERT weights for relation-only classes
    |— lstm_model_no_releation_filtered.pt  # Pre-trained LSTM weights for relation classes
    |— lstm_model.pt           # Pre-trained LSTM weights for all relations
    |— svm pickle files        # Pickle files for SVM pre-trained model
|— notebooks                   # This folder contains implementation notebooks for training
    |— SVM.ipynb
    |— BERT.ipynb
    |— LSTM.py
    |— LSTM._relation_only.py

```

## Installation

1. Download the project files
2. Open the project folder in your preferred IDE
3. Install the required packages:
   ```
   pip install nltk torch transformers imblearn scipy sklearn
   ```
4. Regarding the model_parameters, please reaching out to me to get the link to download the files

## Usage

1. Run the `user_interface.ipynb` notebook in your IDE to interact with the models
2. Alternatively, you can run `choose_model_test.py` directly to make predictions

## Results Summary

### Performance on Full Dataset (with no_relation class)

| Model | Weighted F1 | Macro F1 |
|-------|-------------|----------|
| LSTM  | 0.79        | 0.36     |
| BERT  | 0.81        | 0.41     |

### Performance on Relation-Only Dataset

| Model | Weighted F1 | Macro F1 |
|-------|-------------|----------|
| LSTM  | 0.82        | 0.61     |
| BERT  | 0.69        | 0.57     |

## Note

If you encounter errors while running the `user_interface.ipynb` notebook, try running `choose_model_test.py` directly to make predictions.
