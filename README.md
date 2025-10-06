# Methods in AI Research - Lab 1

Project for dialog act classification.

# Getting Started
Before running any code, please install the right packages by running `pip install -r requirements.txt`. Please use Python version 3.12.1
<br>To run train, inference and evaluation, please check the instructions in the respective sections.

# Dialog Management System
The Dialog Management System is located in `dialog_agent.py`.  
To run the dialog management system, run that file or use `python dialog_agent.py`.  
An example working conversation can be found in the [Example conversation](#example-conversation) section below.
By default it uses a decision tree model saved in `saved_models/decision_tree` to classify the dialog acts based on utterances, and `datasets/restaurant_info.cvs` to get a dataset of restaurants the system could suggest. The model the dialog manager can be changed to any of the folder names present in the `saved_models` folder. Also `extract_keywords()` from `keyword_extractor.py` is used to extract usefull information from the user utterances. 
<br><br>
If you want to see what information the system saves and what states it transitions into, set `debug_mode=True`, default: `debug_mode=False` when making a new instance of the dialogAgent class. Example: `agent = dialogAgent(model_path='saved_models/decision_tree', debug_mode=True)`. This can be changed at the bottom of the `dialog_agent.py` file. 
<br>When `debug_mode=True` the user can give `exit` as user utterance to stop the dialog manager.

## How to use Configurable Features (`config.json`)
- `config.json` in the project root controls runtime behaviour flags for the dialog agent.
- Each option is a boolean; set to `true` to enable the corresponding behaviour and `false` to disable it.
- Update the file manually or run `python -m json.tool config.json` afterwards to verify the JSON stays valid.
- Configurable Features:
  - `Ask confirmation for each preference or not`: when `true`, the agent confirms every extracted preference with the user. Use utterances such as 'yes' or 'no' when conformation is asked. 
    - Example of when the system asks for conformation: <br>`system: You chose price cheap. Is that correct?`
  - `Allow dialog restarts or not`: when `true`, the agent allows the conversation to restart if the user requests it. Use utterances such as 'let's start over' or just 'restart' 
  - `Informal language instead of formal`: when `true`, responses use informal phrasing; switch to `false` for formal language.
  - `Preferences are asked in random order`: when `true`, the agent randomises the order in which missing preferences are asked.

# Baseline Models
The main code for the baseline models can be found in `baseline_models_code`.
<br>
`baseline_ui.py` is a file with the code to run the UI for the baseline models. With this UI you can test the models on a file with utterances or test the rule-based baseline model by giving your own utterances and the model will classify it.

## Quick Start
Run `baseline_ui.py`.<br>
This will start a Terminal UI with three input options:
- `file`: Run both baseline models on `datasets/dialog_acts.dat` and shows metric scores.
- `try me`: User inputs utterance for the Rule Based Baseline model to classify and gives prediction on given utterance.
- `exit`: Stop script.

## Files 
- `baseline_models_code/.`
  - `main.py`: Main file where functions can be found that prepare the data for baseline models, execute predictions and calculate metrics for evalutation of the baseline models. 
  - `baseline_inform.py`: Class file for the Baseline model that always classifies utterances as 'inform'.
  - `baseline_rulebased.py`: Class file for the Baseline model that classifies utterances based on keyword.
  - `baseline_difficult_cases.py`: A file with one function that is used to test a difficult cases on both baseline models. In the file the function is called twice, ones for every difficult case file.
- `baseline_ui.py`: File with all the code for the UI. This file uses functions from `baselein_models_code/main.py`. By default the file that is tested is, `datasets/dialog_acts.dat`. However this can be changed to any other file that is structered where the first word of every line is the dialog act label and the rest of the line is the utterance. To change this, change the `file_path` value at line 64.

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

## Example conversations
### 1
```bash
$ python dialog_agent.py 
system: Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?
user: hello
system: What part of town do you have in mind?
user: id like the south part of town
system: How pricey would you like the restaurant to be?
user: cheap
system: What kind of food would you like?
user: italian
system: Do you have any additional preferences?
user: no
system: la margherita is the only restaurant that meet your requirements.  What information would you like on this restaurant - phone, address or postcode?
user: address
system: The address of la margherita is 15 magdalene street city centre
user: thank you!
```
### 2
```bash
$ python dialog_agent.py 
system: Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?
user: hello
system: What part of town do you have in mind?
user: id like cheap italian food from the south part of town
system: Do you have any additional preferences?
user: no
system: la margherita is the only restaurant that meet your requirements.  What information would you like on this restaurant - phone, address or postcode?
user: phone
system: The phone number of restaurant la margherita is 01223 315232
user: thank you!
```

## Notes
- The training pipeline drops labels with fewer than two samples so stratified splitting remains valid.
- Logistic regression uses `solver='saga'` with `class_weight='balanced'` to handle label imbalance.
- 
