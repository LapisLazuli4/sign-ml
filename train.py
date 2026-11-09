"""
    TRAIN.PY TRAINING THE MODEL

Saves the final model (trained on everyone) to isl_model.pkl.
"""
import argparse
import glob
import pickle

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import FEATURE_LENGTH, WINDOW_FRAMES

FEATURE_COLS = [f"f{i}" for i in range(FEATURE_LENGTH)]
MODEL_PATH = "isl_model.pkl"


def load_data():
    files = sorted(glob.glob("data/*.csv"))
    if not files:
        raise SystemExit("No CSV files found in data/. Run collect.py first.")
    print("Loading:", ", ".join(files))
    frames = [pd.read_csv(f, dtype={"label": str, "contributor": str}) for f in files]
    df = pd.concat(frames, ignore_index=True)

    if f"f{FEATURE_LENGTH - 1}" not in df.columns:
        raise SystemExit(
            "These CSV files were recorded with an older version of collect.py "
            "(one frame per row). Delete them and record again with the current collect.py."
        )
    return df


def make_model():
    return RandomForestClassifier(
        n_estimators=60,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-contributor", default=None)
    args = parser.parse_args()

    df = load_data()
    print(f"\nTotal samples: {len(df)}")
    print("\nSamples per sign:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nSamples per contributor:")
    print(df["contributor"].value_counts().to_string())

    contributors = sorted(df["contributor"].unique())
    test_name = args.test_contributor or (contributors[0] if len(contributors) > 1 else None)

    if test_name is not None:
        test_df = df[df["contributor"] == test_name]
        train_df = df[df["contributor"] != test_name]
        print(f"\nHeld-out test contributor: {test_name} ({len(test_df)} samples)")

        model = make_model()
        model.fit(train_df[FEATURE_COLS], train_df["label"])
        preds = model.predict(test_df[FEATURE_COLS])

        labels = sorted(df["label"].unique())
        print(f"\nAccuracy on unseen contributor: {accuracy_score(test_df['label'], preds):.1%}")
        print("\nPer-sign report:")
        print(classification_report(test_df["label"], preds, labels=labels, zero_division=0))
        print("Confusion matrix (rows = true sign, columns = predicted sign):")
        print(pd.DataFrame(confusion_matrix(test_df["label"], preds, labels=labels),
                           index=labels, columns=labels).to_string())
    else:
        print("\nOnly one contributor found, so there is no honest held-out test yet.")

    final_model = make_model()
    final_model.fit(df[FEATURE_COLS], df["label"])

    bundle = {
        "model": final_model,
        "sklearn_version": sklearn.__version__,
        "feature_length": FEATURE_LENGTH,
        "window_frames": WINDOW_FRAMES,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved {MODEL_PATH} with signs: {list(final_model.classes_)}")


if __name__ == "__main__":
    main()
