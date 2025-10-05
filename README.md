# Methods in AI Research - Lab 1

Project for dialog act classification.

# Getting Started
Before running any code, please install the right packages by running `pip install -r requirements.txt`. Please use Python version 3.12.1
To run train, inference and evaluation, please check the instructions in the respective sections.

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

## Prerequisites
- Run all commands from the project root.
- Install dependencies once with `pip install -r requirements.txt`.

## Training (`train.py`)
- Default run (logistic regression on the deduplicated dataset): `python train.py`
- Switch model: `python train.py --model decision_tree`
- Use a different dataset: `python train.py --data datasets/<your_file.dat>`
- Persist artifacts for later inference/evaluation: `python train.py --save-dir saved_models/<folder_name>`

Example – retrain the bundled decision-tree model on the deduplicated data:

```bash
python train.py --model decision_tree --data datasets/dialog_acts_deduplicated.dat --save-dir saved_models_new/decision_tree
```

Training prints accuracy, a classification report, and a confusion matrix. When `--save-dir` is provided, it creates `model.joblib`, `vectorizer.joblib`, `label_encoder.joblib`, and `metadata.json` inside the target folder.

## Saved Artifacts
- The repository already ships with models trained on the deduplicated dataset in `saved_models/logistic_regression` and `saved_models/decision_tree`.
- Directories without the `_original` suffix use the deduplicated data; the `_original` variants were trained on the raw dataset.
- Each directory contains the three `.joblib` files required by `infer.py` plus `metadata.json`.

## Inference (`infer.py`)
Use `--model-dir` to point at a directory containing saved artifacts and provide input via `--input` (single utterance) or `--file` (one utterance per line).

```bash
# Classify a single utterance with the deduplicated logistic regression model
python infer.py --model-dir saved_models/logistic_regression --input "book a table for two in rome"

# Batch inference with probabilities using the deduplicated decision tree. Use the --file argument to run inference on a file containing multiple utterances
python infer.py --model-dir saved_models/decision_tree --file datasets/samples.txt
```

The script echoes each utterance with its predicted dialog act. When `--proba` is set and the model supports probabilities, the top-k class confidences are printed.

## Evaluation (`evaluate.py`)
`evaluate.py` rebuilds the same stratified test split used during training (default dataset: `datasets/dialog_acts_deduplicated.dat`). Point it at the saved models or baselines you want to score.

```bash
# Evaluate the deduplicated logistic regression model
python evaluate.py --model logistic_regression

# Evaluate the deduplicated decision tree model
python evaluate.py --model decision_tree

# Evaluate both deduplicated saved models at once
python evaluate.py --model saved

# Evaluate a custom artifact directory
python evaluate.py --model saved_models/decision_tree --data datasets/dialog_acts_deduplicated.dat

# Evaluation for the basline models:  

# Score the keyword and "always inform" baselines
python evaluate.py --model baseline

# Evaluate a single baseline by key
python evaluate.py --model baseline_rulebased

```

Each run prints accuracy, a full classification report, and a confusion matrix for the selected models.

## Reference: Datasets and Helpers
- `train.py`: Training CLI supporting `--model {logistic_regression,decision_tree}` and `--data <path>`.
- `infer.py`: Loads saved artifacts to predict labels for inputs from `--input` or `--file`.
- `evaluate.py`: Scores saved or baseline models; accepts `--model`, `--data`, and `--artifacts-root`.
- `datasets/`: contains
  - `dialog_acts.dat`: original dataset (label + utterance per line)
  - `dialog_acts_lower.dat`: lowercased version of the dataset
  - `dialog_acts_deduplicated.dat`: deduplicated dataset used by default in scripts
  - `difficult_cases_multiple.dat`: multiple dialog acts in a single utterance
  - `difficult_cases_negation.dat`: negation-heavy utterances
- `preprocess_dataset/`: helpers for loading, stratified splitting, vectorising, and label encoding
- `utils.py`: CLI utilities for lowercasing, deduplicating, plotting label counts, and auditing duplicates; run `python utils.py --help` to explore the commands

## Notes
- The training pipeline drops labels with fewer than two samples so stratified splitting remains valid.
- Logistic regression uses `solver='saga'` with `class_weight='balanced'` to handle label imbalance.
