import numpy as np
import pandas as pd


class EPLDataPreprocessor:
    """Modular preprocessing pipeline for the EPL FC 24 player dataset."""

    CORE_COLS = ["age", "overall", "potential", "wage_eur"]
    OUT_COLS = [
        "player_id", "short_name", "club_name", "primary_pos", "pos_group",
        "age", "overall", "potential", "growth_gap", "age_curve",
        "is_DEF", "is_MID", "is_FWD", "wage_eur", "log_wage",
    ]

    def __init__(
        self,
        input_path: str = "epl_players_fc24.csv",
        output_path: str = "epl_players_preprocessed.csv",
    ):
        self.input_path = input_path
        self.output_path = output_path

    @staticmethod
    def _map_pos_group(pos: str) -> str:
        if pos == "GK":
            return "GK"
        if pos in {"CB", "RB", "LB", "LWB", "RWB"}:
            return "DEF"
        if pos in {"CM", "CDM", "CAM", "RM", "LM"}:
            return "MID"
        return "FWD"

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.input_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.drop_duplicates(subset=["player_id"]).dropna(subset=self.CORE_COLS).copy()
        return cleaned[cleaned["wage_eur"] > 0].copy()

    def encode_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["primary_pos"] = df["player_positions"].astype(str).str.split(",").str[0].str.strip()
        df["pos_group"] = df["primary_pos"].apply(self._map_pos_group)
        df["is_DEF"] = (df["pos_group"] == "DEF").astype(int)
        df["is_MID"] = (df["pos_group"] == "MID").astype(int)
        df["is_FWD"] = (df["pos_group"] == "FWD").astype(int)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["growth_gap"] = df["potential"] - df["overall"]
        df["age_curve"] = (df["age"] - 27) ** 2
        return df

    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["log_wage"] = np.log(df["wage_eur"]).round(6)
        return df

    def export_data(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df[self.OUT_COLS].copy()
        out_df["wage_eur"] = out_df["wage_eur"].astype(int)
        out_df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(out_df)} preprocessed rows to {self.output_path}")
        return out_df

    def run(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.clean_data(df)
        df = self.encode_positions(df)
        df = self.engineer_features(df)
        df = self.transform_target(df)
        return self.export_data(df)


if __name__ == "__main__":
    EPLDataPreprocessor().run()
