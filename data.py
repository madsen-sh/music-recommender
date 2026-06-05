import numpy as np
import pandas as pd

np.random.seed(42)

GENRES = ["Pop", "Hip-Hop", "Rock", "Electronic", "Jazz", "R&B", "Classical", "Latin"]

ARTISTS = [
    "Nova Skies", "The Drift", "Luna Park", "Cero Sol", "Static Blue",
    "Velvet Hour", "Midnight Fern", "Echo Plains", "Coral Haze", "The Pines",
    "Ivory Fade", "Dark Tempo", "Glass River", "Sunken Feel", "Prism Falls",
    "Hollow Beat", "The Verge", "Amber Lux", "Cold Current", "Silver Haze"
]


def generate_songs(n_songs=500):

    genres = np.random.choice(GENRES, size=n_songs)
    artists = np.random.choice(ARTISTS, size=n_songs)
    tempo = np.random.normal(loc=120, scale=30, size=n_songs).clip(60, 200)

    energy = np.random.beta(a=2, b=2, size=n_songs)

    danceability = np.random.beta(a=2, b=2, size=n_songs)

    valence = np.random.beta(a=2, b=2, size=n_songs)

    acousticness = np.random.beta(a=1.5, b=3, size=n_songs)

    instrumentalness = np.random.beta(a=1, b=5, size=n_songs)

    loudness = np.random.normal(loc=-8, scale=5, size=n_songs).clip(-40, 0)

    duration_sec = np.random.normal(loc=210, scale=45, size=n_songs).clip(90, 420)
    popularity = np.random.randint(0, 101, size=n_songs)

    genre_energy_boost = {
        "Rock": 0.2, "Electronic": 0.25, "Hip-Hop": 0.15,
        "Pop": 0.05, "Jazz": -0.1, "Classical": -0.2,
        "R&B": 0.0, "Latin": 0.1
    }

    for i, genre in enumerate(genres):
        boost = genre_energy_boost[genre]
        energy[i] = np.clip(energy[i] + boost, 0, 1)
        danceability[i] = np.clip(danceability[i] + boost * 0.5, 0, 1)

        song_ids = [f"SONG_{str(i).zfill(4)}" for i in range(n_songs)]

    df = pd.DataFrame({
        "song_id":          song_ids,
        "title":            [f"Track {i+1}" for i in range(n_songs)],
        "artist":           artists,
        "genre":            genres,
        "tempo":            np.round(tempo, 2),
        "energy":           np.round(energy, 4),
        "danceability":     np.round(danceability, 4),
        "valence":          np.round(valence, 4),
        "acousticness":     np.round(acousticness, 4),
        "instrumentalness": np.round(instrumentalness, 4),
        "loudness":         np.round(loudness, 2),
        "duration_sec":     np.round(duration_sec, 1),
        "popularity":       popularity,
    })

    return df

if __name__ == "__main__":
    df = generate_songs()
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 3 songs:")
    print(df.head(3).to_string())
    print(f"\nGenre distribution:")
    print(df["genre"].value_counts())

