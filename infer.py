import argparse
from pathlib import Path
import sys
import json
import joblib


def load_artifacts(model_dir: Path):
    """Load serialized inference assets from `model_dir`.

    Inputs:
        model_dir: Directory containing `model.joblib`, `vectorizer.joblib`,
            `label_encoder.joblib`, and optional `metadata.json`.
    Returns:
        Tuple of (model, vectorizer, label_encoder, metadata dict).
    """
    model = joblib.load(model_dir / "model.joblib")
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    meta_path = model_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, vectorizer, label_encoder, metadata


def read_inputs(args):
    """Collect inference utterances from CLI flags or a text file.

    Inputs:
        args: Parsed argparse Namespace with `input` and/or `file` attributes.
    Returns:
        List of non-empty utterance strings ready for vectorization.
    """
    texts = []
    if args.input:
        texts.append(args.input)
    if args.file:
        with open(args.file, "r") as f:
            texts.extend([ln.strip() for ln in f if ln.strip()])
    if not texts:
        print("No input provided. Use --input or --file.")
        sys.exit(1)
    return texts

def infer_utterance(model, vectorizer, label_encoder, metadata, utterance):
    """Predict a dialog-act label for a single `utterance`.

    Inputs:
        model: Trained classifier exposing `predict`.
        vectorizer: Text vectorizer implementing `transform`.
        label_encoder: Encoder used to map prediction indices to labels.
        metadata: Optional dictionary with model metadata (unused here).
        utterance: Single text string to classify.
    Returns:
        Predicted dialog-act label as a string.
    """
    if utterance is None:
        raise ValueError("utterance must be provided")

    # Vectorize the single utterance and decode the predicted class label
    features = vectorizer.transform([utterance])
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]

    return str(label)

def main():
    """Entry point for running offline inference with a saved model.

    Inputs:
        None directly; reads CLI flags for model directory and inputs.
    Returns:
        None; prints predicted labels (and optional probabilities) to stdout.
    """
    parser = argparse.ArgumentParser(description="Infer dialog acts using a saved model")
    parser.add_argument("--model-dir", required=True, help="Directory containing saved artifacts")
    parser.add_argument("--input", help="Single utterance to classify")
    parser.add_argument("--file", help="Path to a file with one utterance per line")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k probabilities if supported")
    parser.add_argument("--proba", action="store_true", help="Print class probabilities")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model, vectorizer, label_encoder, metadata = load_artifacts(model_dir)

    texts = read_inputs(args)
    X = vectorizer.transform(texts)

    preds = model.predict(X)
    labels = label_encoder.inverse_transform(preds)

    # output predictions
    for i, (text, lab) in enumerate(zip(texts, labels), 1):
        print(f"[{i}] {lab}\t{text}")

    # optional probabilities
    if args.proba and hasattr(model, "predict_proba"):
        print("\nProbabilities:")
        probas = model.predict_proba(X)
        classes = list(label_encoder.classes_)
        for i, p in enumerate(probas, 1):
            # top-k sorted per example
            top = sorted(zip(classes, p), key=lambda t: t[1], reverse=True)[: args.topk]
            top_str = ", ".join([f"{c}: {score:.3f}" for c, score in top])
            print(f"[{i}] {top_str}")


if __name__ == "__main__":
    main()
