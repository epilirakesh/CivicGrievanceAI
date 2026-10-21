from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "astra_baseline_model_122.joblib"
_model = joblib.load(MODEL_PATH)

def predict_complaint(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Complaint text cannot be empty.")

    text = text.strip()
    category = str(_model.predict([text])[0])
    probabilities = _model.predict_proba([text])[0]

    scores = {
        str(label): round(float(score), 4)
        for label, score in zip(_model.classes_, probabilities)
    }

    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]
    needs_review = margin < 0.05

    return {
        "category": category,
        "confidence": scores[category],
        "all_scores": scores,
        "score_margin": round(margin, 4),
        "needs_review": needs_review
    }

if __name__ == "__main__":
    example = "Road par bahut bade potholes hain."
    print(predict_complaint(example))
