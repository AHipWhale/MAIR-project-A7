# uu_p1_methods_in_ai

Project for dialog act classification with simple data prep utilities and a training CLI.

# Getting Started
Before running any code, please install the right packages by running `pip install -r requirements.txt`.

# Dialog Management System
The Dialog Management System is located in `dialog_agent.py`. To run the dialog management system, run that file. By default it uses a decision tree model saved in `artifacts/dt` to classify the dialog acts based on utterances, and `datasets/restaurant_info.cvs` to get a dataset of restaurants the system could suggest. Also `extract_keywords()` from `keyword_extractor.py` is used to extract usefull information from the user utterances. 
<br><br>
If you want to see what information the system saves and what states it transitions into, set `debug_mode=True`, default: `debug_mode=False` when making a new instance of the dialogAgent class. Example: `agent = dialogAgent(model_path='artifacts/dt', debug_mode=True)`

# Baseline Models
All code for this part can be found in `baseline_models_code`.

## Quick Start
Run `baseline_models_code/main.py`.<br>
This will start a Terminal UI with three input options:
- `file`: Run both baseline models on `datasets/dialog_acts.dat` and shows metric scores.
- `try me`: User inputs utterance for the Rule Based Baseline model to classify and gives prediction on given utterance.
- `exit`: Stop script.

## Files (`baseline_models_code/.`)
- `main.py`: Main file where data is prepared for baseline models, predictions are executed and terminal UI code can be found.
- `baseline_inform.py`: Class file for the Baseline model that always classifies utterances as 'inform'.
- `baseline_rulebased.py`: Class file for the Baseline model that classifies utterances based on keyword.

# Machine Learning models

## Quick Start
- Default (logistic regression): `python train.py`
- Select model: `python train.py --model decision_tree`
- Select dataset: `python train.py --data datasets/dialog_acts_lower.dat`

### Save and Infer
- Save artifacts while training: `python train.py --save-dir artifacts/logreg`
- Run inference on a single utterance: `python infer.py --model-dir artifacts/logreg --input "book a flight to rome"`
- Run inference on a file (one utterance per line): `python infer.py --model-dir artifacts/logreg --file my_utterances.txt`
- Show top-k probabilities: `python infer.py --model-dir artifacts/logreg --file my_utterances.txt --proba --topk 5`

## Top-Level Files
- `train.py`: CLI to train a classifier on the dataset. Supports `--model {logistic_regression,decision_tree}` and `--data <path>`. Prints accuracy, classification report, and confusion matrix.
- `infer.py`: CLI to load saved artifacts (`model.joblib`, `vectorizer.joblib`, `label_encoder.joblib`) and predict labels for inputs.
- `datasets/`: Folder containing dataset files:
  - `dialog_acts.dat`: Original dataset (label + utterance per line).
  - `dialog_acts_lower.dat`: Lowercased version of the dataset.
  - `dialog_acts_deduplicated.dat`: Deduplicated dataset (removed duplicate utterances).
  - `difficult_cases_multiple.dat`: Dataset with difficult cases where multiple dialog acts are present in one utterance.
  - `difficult_cases_negation.dat`: Dataset with difficult cases that contain negation but are different dialog acts.

## Data Prep Module (`preprocess_dataset/`)
- `__init__.py`: Re-exports helpers for convenient import.
- `dataio.py`: Loads a space-separated `label utterance` file into a pandas DataFrame with columns `label` and `text`.
- `split.py`: Creates a stratified train/test split (default 85/15) using scikit-learn.
- `vectorize.py`: Fits a `CountVectorizer` on train text and transforms train/test.
- `encode.py`: Encodes labels using `LabelEncoder` (fit on train, transform train/test).
- `prepare.py`: Orchestrates loading, filtering labels with <2 samples, splitting, vectorizing, encoding, and prints a brief dataset summary.

## Utility Scripts (`utils/`)
- `convert_data_to_lowercase.py`: Lowercases labels and utterances in a dataset file.
- `remove_duplicates.py`: Removes duplicate lines (keeps first occurrence).
- `get_label_dist.py`: Prints label distribution statistics.
- `check_more_than_one_dialog_act.py`: Checks for lines containing more than one dialog act token.

## Notes
- The training pipeline drops labels with fewer than 2 samples to keep stratified splitting valid.
- For multiclass text data, `solver='saga'` is used with LogisticRegression and `class_weight='balanced'` to handle imbalance.
....