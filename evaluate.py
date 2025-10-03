import argparse
import os
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess_dataset import load_data_to_df, stratified_split

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASELINE_CODE_DIR = Path(__file__).resolve().parent / "baseline_models_code"
if str(BASELINE_CODE_DIR) not in sys.path:
    sys.path.append(str(BASELINE_CODE_DIR))

from baseline_inform import BaselineInform  # noqa: E402
from baseline_rulebased import BaselineRuleBased  # noqa: E402
from main import calc_metrics  # noqa: E402

DEFAULT_DATASET = "datasets/dialog_acts_deduplicated.dat"
DEFAULT_ARTIFACTS_ROOT = Path("saved_models")

SAVED_MODEL_KEYS = {"logistic_regression", "decision_tree"}
BASELINE_MODEL_KEYS = {"baseline_inform", "baseline_rulebased"}


def load_test_split(data_path: Path):
    """Recreate the deterministic test split used during training."""
    df = load_data_to_df(data_path)
    label_counts = df["label"].value_counts()
    rare_labels = label_counts[label_counts < 2].index.tolist()

    major_df = df[~df["label"].isin(rare_labels)]
    if major_df.empty:
        raise ValueError("No samples available for stratified split; dataset contains only rare labels.")

    _, x_test, _, y_test = stratified_split(major_df, test_size=0.15, seed=42)
    return x_test, y_test


def print_saved_model_metrics(model_name: str, y_true, y_pred, label_encoder) -> None:
    print("\n" + "-" * 150)
    print(f"Metric scores of model: {model_name}")
    print("\nAccuracy:", accuracy_score(y_true, y_pred), "\n")
    labels_idx = list(range(len(label_encoder.classes_)))
    print(
        "Classification Report:\n",
        classification_report(
            y_true,
            y_pred,
            labels=labels_idx,
            target_names=label_encoder.classes_,
            zero_division=0,
        ),
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    df_cm = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("Confusion Matrix (counts)")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df_cm.to_string())


def evaluate_saved_model(
    model_key: str,
    artifacts_root: Path,
    x_test_text,
    y_test_labels,
    artifacts_dir: Path | None = None,
) -> None:
    if artifacts_dir is None:
        artifacts_dir = artifacts_root / model_key
    else:
        artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found for '{model_key}': {artifacts_dir}")

    classifier = joblib.load(artifacts_dir / "model.joblib")
    vectorizer = joblib.load(artifacts_dir / "vectorizer.joblib")
    label_encoder = joblib.load(artifacts_dir / "label_encoder.joblib")

    x_test_vectorized = vectorizer.transform(x_test_text)
    y_true_encoded = label_encoder.transform(y_test_labels)
    y_pred_encoded = classifier.predict(x_test_vectorized)

    display_name = model_key.replace("_", " ").title() if model_key in SAVED_MODEL_KEYS else artifacts_dir.name
    print_saved_model_metrics(display_name, y_true_encoded, y_pred_encoded, label_encoder)


def evaluate_baseline(model_key: str, x_test_text, y_test_labels) -> None:
    if model_key == "baseline_inform":
        predictor = BaselineInform()
        display_name = "Baseline Inform"
    elif model_key == "baseline_rulebased":
        predictor = BaselineRuleBased()
        display_name = "Baseline keywords"
    else:
        raise ValueError(f"Unknown baseline model key: {model_key}")

    y_pred = predictor.predict(x_test_text)
    calc_metrics(display_name, y_test_labels, y_pred)


def expand_model_selection(selection: str):
    if selection == "all":
        return ["logistic_regression", "decision_tree", "baseline_inform", "baseline_rulebased"]
    if selection == "saved":
        return ["logistic_regression", "decision_tree"]
    if selection == "baseline":
        return ["baseline_inform", "baseline_rulebased"]
    if selection in SAVED_MODEL_KEYS | BASELINE_MODEL_KEYS:
        return [selection]

    potential_path = Path(selection)
    if potential_path.exists():
        return [potential_path]

    raise ValueError(
        "Unsupported model selection: {sel}. Expected one of built-in keys or a valid artifacts directory path.".format(
            sel=selection
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved or baseline dialog-act models on the test split used during training.")
    parser.add_argument(
        "-m",
        "--model",
        default="all",
        help=(
            "Which model(s) to evaluate. Use built-in keys (all, saved, baseline, logistic_regression, "
            "decision_tree, baseline_inform, baseline_rulebased) or provide a path to a directory "
            "containing saved artifacts."
        ),
    )
    parser.add_argument(
        "-d",
        "--data",
        default=DEFAULT_DATASET,
        help=f"Dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(DEFAULT_ARTIFACTS_ROOT),
        help="Root directory containing saved model artifacts (default: saved_models)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    artifacts_root = Path(args.artifacts_root)
    x_test_text, y_test_labels = load_test_split(data_path)

    for model_key in expand_model_selection(args.model):
        if isinstance(model_key, Path):
            evaluate_saved_model(
                model_key.name,
                artifacts_root,
                x_test_text,
                y_test_labels,
                artifacts_dir=model_key,
            )
        elif model_key in SAVED_MODEL_KEYS:
            evaluate_saved_model(model_key, artifacts_root, x_test_text, y_test_labels)
        elif model_key in BASELINE_MODEL_KEYS:
            evaluate_baseline(model_key, x_test_text, y_test_labels)
        else:
            raise ValueError(f"Unsupported model selection: {model_key}")


if __name__ == "__main__":
    main()
