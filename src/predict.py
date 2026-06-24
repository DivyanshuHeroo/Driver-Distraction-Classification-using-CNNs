"""Single-image inference + risk assessment.

Used by both the CLI and the Streamlit dashboard. Wraps the model so the rest
of the app only deals with a clean dictionary result.

Run (CLI):
    python -m src.predict --image path/to/img.jpg \
        --model models/driver_distraction_model.keras
"""

import os
import argparse

import numpy as np
import tensorflow as tf

from src.utils import load_config, get_class_names
from src.preprocess import load_and_preprocess, make_batch

# Recommended action shown to the user per risk level.
RISK_ACTIONS = {
    "SAFE": "No action needed. Driver is attentive.",
    "MEDIUM": "Gentle reminder / dashboard alert recommended.",
    "HIGH": "Immediate audible alert — high-risk distraction detected.",
}


class DistractionPredictor:
    """Loads a model once and serves predictions for individual images."""

    def __init__(self, config_path="config.yaml", model_path=None):
        self.config = load_config(config_path)
        self.class_names = get_class_names(self.config)
        self.labels = self.config["classes"]
        self.risk_levels = self.config["risk_levels"]
        self.image_size = tuple(self.config["data"]["image_size"])

        if model_path is None:
            model_path = os.path.join(
                self.config["paths"]["models_dir"],
                "driver_distraction_model.keras",
            )
        self.model = tf.keras.models.load_model(model_path)

    def predict_array(self, image_array):
        """Predict from a preprocessed [0,1] image array of model input size."""
        batch = make_batch(image_array)
        probs = self.model.predict(batch, verbose=0)[0]
        return self._format(probs)

    def predict_file(self, image_path):
        """Predict directly from an image file on disk."""
        arr = load_and_preprocess(image_path, self.image_size)
        return self.predict_array(arr)

    def _format(self, probs):
        idx = int(np.argmax(probs))
        class_code = self.class_names[idx]
        risk = self.risk_levels[class_code]
        return {
            "class_code": class_code,
            "label": self.labels[class_code],
            "confidence": float(probs[idx]),
            "risk_level": risk,
            "recommended_action": RISK_ACTIONS[risk],
            "probabilities": {
                self.labels[self.class_names[i]]: float(probs[i])
                for i in range(len(probs))
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Predict driver distraction")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    predictor = DistractionPredictor(args.config, args.model)
    result = predictor.predict_file(args.image)

    print(f"\nPredicted : {result['label']} ({result['class_code']})")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print(f"Risk level: {result['risk_level']}")
    print(f"Action    : {result['recommended_action']}")


if __name__ == "__main__":
    main()
