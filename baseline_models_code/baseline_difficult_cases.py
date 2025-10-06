from main import predict_data

def baseline_difficult_cases(difficult_cases_path: str):
    """Classifies utterances in `difficult_cases_path` using both baseline models.
    
    Inputs:
        difficult_cases_path: Path to file with utterances to be classified. 
    
    Returns:
        None: prints results of callification from both baseline models.
    """
    diff_case_labels = []
    diff_case_sentences = []

    # Read, prepare and split file content
    with open(difficult_cases_path) as file:
        for line in file:
            diff_case_prepped_line = line.strip().lower().split(" ", maxsplit=1)
            diff_case_labels.append(diff_case_prepped_line[0])
            diff_case_sentences.append(diff_case_prepped_line[1])

    # Classify utterances using both baseline models
    pred_diff_case_baseInf, pred_diff_case_baseRule = predict_data(diff_case_sentences)

    # Print out results
    print(f"Results for '{difficult_cases_path}' with baseline Inform:")
    for pred_item in enumerate(pred_diff_case_baseInf):
        print(pred_item)

    print(f"\nResults for '{difficult_cases_path}' with baseline RuleBased:")
    for pred_item in enumerate(pred_diff_case_baseRule):
        print(pred_item)
    
    print("\n"+"-"*150)

# Run classification for both difficult cases files
baseline_difficult_cases("./datasets/difficult_cases_multiple.dat")
baseline_difficult_cases("./datasets/difficult_cases_negation.dat")