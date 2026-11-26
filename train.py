import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FairWageTrainer:
    """Modular ML training pipeline to predict fair weekly wages for EPL players."""

    FEATURES = [
        "overall",
        "potential",
        "age",
        "age_curve",
        "is_DEF",
        "is_MID",
        "is_FWD",
    ]
    TARGET = "log_wage"

    def __init__(
        self,
        input_path: str = "epl_players_preprocessed.csv",
        scored_output_path: str = "epl_players_scored.csv",
        weights_output_path: str = "model_weights.json",
        test_size: float = 0.2,
        random_state: int = 42,
        alpha: float = 1.0,
    ):
        self.input_path = input_path
        self.scored_output_path = scored_output_path
        self.weights_output_path = weights_output_path
        self.test_size = test_size
        self.random_state = random_state
        self.alpha = alpha

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.input_path)

    def split_data(self, df: pd.DataFrame):
        X = df[self.FEATURES]
        y = df[self.TARGET]
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=self.alpha)),
        ])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred_log = pipeline.predict(X_test)
        y_true_eur = np.exp(y_test)
        y_pred_eur = np.exp(y_pred_log)

        metrics = {
            "r2_log": round(float(r2_score(y_test, y_pred_log)), 4),
            "r2_eur": round(float(r2_score(y_true_eur, y_pred_eur)), 4),
            "mae_eur": round(float(mean_absolute_error(y_true_eur, y_pred_eur)), 2),
            "test_samples": int(len(y_test)),
        }
        print(
            f"Evaluation -> R2 (log): {metrics['r2_log']} | "
            f"R2 (EUR): {metrics['r2_eur']} | MAE: EUR {metrics['mae_eur']:,.0f}/week"
        )
        return metrics

    @staticmethod
    def _label_status(ratio: float) -> str:
        if ratio >= 1.25:
            return "Overpaid"
        if ratio <= 0.80:
            return "Underpaid"
        return "Fair"

    def score_players(self, pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
        scored = df.copy()
        pred_log = pipeline.predict(df[self.FEATURES])
        scored["fair_wage_eur"] = np.round(np.exp(pred_log)).astype(int)
        scored["wage_diff_eur"] = scored["wage_eur"] - scored["fair_wage_eur"]
        scored["wage_ratio"] = (scored["wage_eur"] / scored["fair_wage_eur"]).round(3)
        scored["wage_status"] = scored["wage_ratio"].apply(self._label_status)
        return scored

    def export_outputs(self, pipeline: Pipeline, scored_df: pd.DataFrame, metrics: dict) -> None:
        scored_df.to_csv(self.scored_output_path, index=False, encoding="utf-8-sig")

        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["model"]
        artifact = {
            "features": self.FEATURES,
            "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
            "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
            "coefficients": [round(float(v), 6) for v in model.coef_],
            "intercept": round(float(model.intercept_), 6),
            "alpha": self.alpha,
            "metrics": metrics,
        }
        with open(self.weights_output_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)

        print(f"Saved scored dataset to {self.scored_output_path} ({len(scored_df)} rows)")
        print(f"Saved model weights to {self.weights_output_path}")

    def run(self) -> tuple[pd.DataFrame, dict]:
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.split_data(df)
        pipeline = self.train(X_train, y_train)
        metrics = self.evaluate(pipeline, X_test, y_test)
        scored_df = self.score_players(pipeline, df)
        self.export_outputs(pipeline, scored_df, metrics)
        return scored_df, metrics


if __name__ == "__main__":
    FairWageTrainer().run()
