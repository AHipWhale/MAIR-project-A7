from main import predict_data

def baseline_difficult_cases(difficult_cases_path: str):
    diff_case_labels = []
    diff_case_sentences = []

    with open(difficult_cases_path) as file:
        for line in file:
            diff_case_prepped_line = line.strip().lower().split(" ", maxsplit=1)
            diff_case_labels.append(diff_case_prepped_line[0])
            diff_case_sentences.append(diff_case_prepped_line[1])

    # start_up_ui(diff_case_sentences, diff_case_labels)
    pred_diff_case_baseInf, pred_diff_case_baseRule = predict_data(diff_case_sentences)

    print(f"Results for '{difficult_cases_path}' with baseline Inform:")
    for pred_item in enumerate(pred_diff_case_baseInf):
        print(pred_item)

    print(f"\nResults for '{difficult_cases_path}' with baseline RuleBased:")
    for pred_item in enumerate(pred_diff_case_baseRule):
        print(pred_item)
    
    print("\n"+"-"*150)
    
baseline_difficult_cases("./datasets/difficult_cases_multiple.dat")
baseline_difficult_cases("./datasets/difficult_cases_negation.dat")