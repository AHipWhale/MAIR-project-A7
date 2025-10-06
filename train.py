import os
import warnings

# Joblib cannot spawn processes in this environment; force serial execution to avoid warnings.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
warnings.filterwarnings("ignore")

from preprocess_dataset import prepare_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import argparse
from pathlib import Path
import json
from datetime import datetime
import joblib


def main():
    """Parse CLI args, train the selected model, and optionally persist artifacts.

    Inputs:
        Reads command-line arguments for model choice, dataset path, and artifact output folder.
    Returns:
        None; prints evaluation metrics and writes artifacts when a save directory is supplied.
    """
    parser = argparse.ArgumentParser(description="Train a model on dialog acts")
    parser.add_argument(
        "-m",
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "decision_tree"],
        help="Model to train (default: logistic_regression)",
    )
    parser.add_argument(
        "-d",
        "--data",
        default="datasets/dialog_acts_deduplicated.dat",
        help="Path to dataset file (default: datasets/dialog_acts_deduplicated.dat)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save trained artifacts (model, vectorizer, encoder). If omitted, nothing is saved.",
    )
    args = parser.parse_args()

    # preprocess dataset and get train/test splits. split is done in a stratified manner
    data = prepare_dataset(args.data)
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    label_encoder = data['encoder']
    vectorizer = data['vectorizer']

    # choose model
    if args.model == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced', C=1.0)
    else:  # decision_tree
        classifier = DecisionTreeClassifier(
            criterion='gini',
            random_state=0,
            ccp_alpha=0.0002,
        )

    # fit the model to the training data
    classifier.fit(x_train, y_train)

    # evaluate on the test data
    pred = classifier.predict(x_test)
    label_indices = list(range(len(label_encoder.classes_)))
    print("\nAccuracy:", accuracy_score(y_test, pred), '\n')
    print(
        "Classification Report:\n",
        classification_report(
            y_test,
            pred,
            labels=label_indices,
            target_names=label_encoder.classes_,
            zero_division=0,
        ),
    )
    # format and print confusion matrix with class labels
    cm = confusion_matrix(y_test, pred, labels=label_indices)
    labels = list(label_encoder.classes_)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion Matrix (counts)")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df_cm.to_string())

    # optionally save artifacts
    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(classifier, out_dir / "model.joblib")
        joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
        joblib.dump(label_encoder, out_dir / "label_encoder.joblib")

        meta = {
            "model_type": args.model,
            "dataset": args.data,
            "classes": list(label_encoder.classes_),
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"\nSaved artifacts to: {out_dir}")

if __name__ == "__main__":
    main()
