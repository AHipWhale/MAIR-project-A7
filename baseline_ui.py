from baseline_models_code.main import predict_data, calc_metrics, test_rule_based
from preprocess_dataset import load_data_to_df, stratified_split

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
    # Path to data file
    file_path = "./datasets/dialog_acts.dat"

    # Load data into pandas dataframe
    df = load_data_to_df(file_path)

    # Split data into train- and testset
    _, x_test, _, y_test = stratified_split(df)

    # Run terminal UI
    start_up_ui(x_test, y_test)