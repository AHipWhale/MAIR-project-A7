# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import models
from baseline_inform import BaselineInform
from baseline_rulebased import BaselineRuleBased

def calc_metrics(model_name: str, y_true: list, y_pred: list):
    """
    Calculate accuracy, precision, recall, f1-score and show a confusion matrix based on 'y_true' and 'y_pred'
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
    """
    Predicts class for data in 'x_test' using the Baseline Inform and Baseline Rule Based models
    """
    # predictions based on Baseline Inform
    baseInform = BaselineInform()
    baseInform_res = baseInform.predict(x_test)

    # predictions based on Baseline Rule Based
    baseRuleBased = BaselineRuleBased()
    baseRuleBased_res = baseRuleBased.predict(x_test)
    return baseInform_res, baseRuleBased_res

def test_rule_based(utterance: str) -> str:
    """
    Predicts class of 'utterance' based on Baseline Rule Based model
    """
    baseRuleBased = BaselineRuleBased()
    baseRuleBased_res = baseRuleBased.predict([utterance])
    return baseRuleBased_res[0]

def start_up_ui(x_test: list, y_test: list):
    """
    Function to run terminal UI
    """
    print("You're running my script! What is it you want to do?")
    print("If you want me to classify the 'dialog_acts.dat' file, type 'file'.")
    print("Or if you want me to classify one of your utterances, type 'try me'.")
    print("If you want to stop this script, type 'exit'.")

    user_input = input("What is it going to be?: ")

    # Input validation
    if user_input not in ['file', 'try me', 'exit']:
        print('You entered in an incorrect input, try again!')
        user_input = input("What is it going to be?: ")
    
    # Run predictions on 'dialog_acts.dat' and show metrics
    if user_input == 'file':
        print('Okay, you want me to show my skills, here you go!')
        baseInform_res, baseRuleBased_res = predict_data(x_test)
        calc_metrics("Baseline Inform", y_test, baseInform_res)
        calc_metrics("Baseline keywords", y_test, baseRuleBased_res)

    # Run predictions on user input utterance
    elif user_input == 'try me':
        print('You want to test my skills!')
        # Loop until user wants to exit
        while True:
            input_utterance = input("Enter an utterance to classify (or type 'exit' to quit): ").lower().strip()
            if input_utterance.lower() == 'exit':
                print("Exiting the program.")
                break
            input_prediction = test_rule_based(input_utterance)
            print(f"The predicted dialog act is: {input_prediction}")

    # Exit the program
    else:
        print("You're shutting me down...")
        exit()

if __name__ == "__main__":
    # Path to 'dialog_acts.dat' file
    file_path = "./datasets/dialog_acts.dat"

    # List for feature and target
    labels = []
    sentences = []

    # Open file
    with open(file_path) as f:
        for line in f:
            # Remove unwanted spaces and characters (like '\n'), convert string to lower case, split the act from the dialog and store as lists in list
            prepped_line = line.strip().lower().split(" ", maxsplit=1)
            labels.append(prepped_line[0])
            sentences.append(prepped_line[1])

    # Split dataset into train- and testset
    _, x_test, _, y_test = train_test_split(sentences, labels, random_state=42, test_size=0.15, stratify=labels)

    # Run terminal UI
    start_up_ui(x_test, y_test)
