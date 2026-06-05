import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

FEATURES_COLS = [
    "tempo",
    "energy",
    "danceability",
    "valence",
    "acousticness",
    "instrumentalness",
    "loudness",
    "duration__Sec",
]

def engineer_features(df):

    df = df.copy()

    df["tempo_normalized"] = (df["tempo"] - 60) / (200 - 60)

    df["loudness_normalized"] = (df["loudness"] - (-40)) / (0 - (-40))

    scaler = MinMaxScaler()

    cols_to_scale = ["energy", "danceability", "valence",
                     "acousticness", "instrumentalness", "duration_sec"]

    scaled_values = scaler.fit_transform(df[cols_to_scale])

    scaled_df = pd.DataFrame(scaled_values,
                             columns=[f"{c}_scaled" for c in cols_to_scale],
                             index=df.index)

    df = pd.concat([df, scaled_df], axis=1)

    feature_matrix = pd.DataFrame({
        "tempo_f":             df["tempo_normalized"],
        "energy_f":            df["energy_scaled"],
        "danceability_f":      df["danceability_scaled"],
        "valence_f":           df["valence_scaled"],
        "acousticness_f":      df["acousticness_scaled"],
        "instrumentalness_f":  df["instrumentalness_scaled"],
        "loudness_f":          df["loudness_normalized"],
        "duration_f":          df["duration_sec_scaled"],
    }, index=df.index)

    return df, feature_matrix


if __name__ == "__main__":
    from data import generate_songs
    
    df_songs = generate_songs()
    df_engineered, feature_matrix = engineer_features(df_songs)


    print(f"Engineered DataFrame shape : {df_engineered.shape}")
    print(f"Feature matrix shape:        {feature_matrix.shape}")
    print(f"\nFeature matrix value sample (first 3 rows):")
    print(feature_matrix.head(3).to_string())
    print(f"\nFeature value ranges (should all be 0.0 to 1.0):")
    print(feature_matrix.describe().loc[["min" , "max"]].to_string())

