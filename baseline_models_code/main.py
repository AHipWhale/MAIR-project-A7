# Import libraries
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import models
from baseline_models_code.baseline_inform import BaselineInform
from baseline_models_code.baseline_rulebased import BaselineRuleBased

def calc_metrics(model_name: str, y_true: list, y_pred: list):
    """Calculate accuracy, precision, recall, f1-score and show a confusion matrix based on 'y_true' and 'y_pred'

    Inputs:
        model_name: Name of the model to show in the output.
        y_true: List of true labels.
        y_pred: List of predicted labels.
    Returns:
        None: prints out metrics to terminal.
    """
    # a line to seperate metrics of different models
    print("\n"+"-"*150)
    
    print(f"Metric scores of model: {model_name}")

    # calculate and print accuracy
    print("\nAccuracy:", accuracy_score(y_true, y_pred))

    labels = ['ack', 'affirm', 'bye', 'confirm', 'deny', 'hello', 'inform', 'negate', 'null', 'repeat', 'reqalts', 'reqmore', 'request', 'restart', 'thankyou']

    # calculate and print precision, recall and f1-score
    print(
        "\nClassification Report:\n",
        classification_report(y_true, y_pred, labels=labels, zero_division=0),
    )

    # calculate and print confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion Matrix (counts)")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df_cm.to_string())

def predict_data(x_test: list) -> tuple:
    """Predicts class for data in 'x_test' using the Baseline Inform and Baseline Rule Based models.

    Inputs:
        x_test: List of utterances to be classified.
    
    Returns:
        baseInform_res: List of predicted labels from Baseline Inform model.
        baseRuleBased_res: List of predicted labels from Baseline Rule Based model.
    """
    # predictions based on Baseline Inform
    baseInform = BaselineInform()
    baseInform_res = baseInform.predict(x_test)

    # predictions based on Baseline Rule Based
    baseRuleBased = BaselineRuleBased()
    baseRuleBased_res = baseRuleBased.predict(x_test)
    return baseInform_res, baseRuleBased_res

def test_rule_based(utterance: str) -> str:
    """Predicts class of 'utterance' based on Baseline Rule Based model.

    Inputs:
        utterance: Utterance to be classified.
    
    Returns:
        Predicted label from Baseline Rule Based model.
    """
    baseRuleBased = BaselineRuleBased()
    baseRuleBased_res = baseRuleBased.predict([utterance])
    return baseRuleBased_res[0]

