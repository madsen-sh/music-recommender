import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity_matrix(feature_matrix):
    similarity_matrix = cosine_similarity(feature_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index = feature_matrix.index,
        columns = feature_matrix.index)
    return similarity_df

def get_recommendations(song_idx, df_songs, similarity_df, n=5):
    if song_idx not in similarity_df.index:
        print(f"SOng Index {song_idx} not found")
        return None

    similarity_scores = similarity_df[song_idx]
    similarity_scores = similarity_scores.drop(index = song_idx)
    top_indices = similarity_scores.nlargest(n).index
    recommendations = df_songs.loc[top_indices].copy()
    
    recommendations["similarity_score"] = similarity_scores[top_indices].values

    recommendations["similarity_pct"] =(
        recommendations["similarity_score"] * 100
    ).round(1)

    return recommendations[["title", "artist", "genre",
                            "energy", "danceability",
                            "valence", "similarity_pct"]]
def explain_recommendation(song_idx, rec_idx, df_songs, feature_matrix):

    song = df_songs.loc[song_idx]
    rec  = df_songs.loc[rec_idx]

    song_features = feature_matrix.loc[song_idx]
    rec_features  = feature_matrix.loc[rec_idx]

    differences = (song_features - rec_features).abs()

    most_similar    = differences.nsmallest(3).index.tolist()
    most_different  = differences.nlargest(3).index.tolist()

    print(f"\n  Why '{rec['title']}' by {rec['artist']} was recommended:")
    print(f"  Most similar on:   {', '.join(most_similar)}")
    print(f"  Most different on: {', '.join(most_different)}")

def build_user_profile(liked_song_indices, feature_matrix):

    liked_features = feature_matrix.loc[liked_song_indices]

    user_profile = liked_features.mean(axis=0)

    return user_profile


def get_personalized_recommendations(user_profile, df_songs,
                                      feature_matrix, liked_indices, n=10):

    user_vector = user_profile.values.reshape(1, -1)

    all_vectors = feature_matrix.values

    scores = cosine_similarity(user_vector, all_vectors)[0]

    scores_series = pd.Series(scores, index=feature_matrix.index)

    scores_series = scores_series.drop(index=liked_indices)

    recommendations = df_songs.copy()
    recommendations["base_score"] = scores_series

    recommendations = recommendations.dropna(subset=["base_score"])

    recommendations["popularity_boost"] = recommendations["popularity"] / 1000

    genre_counts = df_songs.loc[liked_indices, "genre"].value_counts(normalize=True)

    def genre_boost(genre):
        return genre_counts.get(genre, 0) * 0.05

    recommendations["genre_boost"] = recommendations["genre"].apply(genre_boost)

    recommendations["final_score"] = (
        recommendations["base_score"] +
        recommendations["popularity_boost"] +
        recommendations["genre_boost"]
    )

    recommendations = recommendations.sort_values("final_score", ascending=False)

    return recommendations.head(n)[["title", "artist", "genre",
                                     "energy", "danceability",
                                     "valence", "popularity",
                                     "final_score"]]

if __name__ == "__main__":
    from data import generate_songs
    from features import engineer_features

    print("Building dataset...")
    df_songs = generate_songs()
    df_engineered, feature_matrix = engineer_features(df_songs)

    print("Building similarity matrix...")
    similarity_df = build_similarity_matrix(feature_matrix)

    print("\n--- SINGLE SONG RECOMMENDATIONS ---")
    test_song_idx = 0
    test_song = df_songs.loc[test_song_idx]
    print(f"Seed song: '{test_song['title']}' by {test_song['artist']} [{test_song['genre']}]")

    recs = get_recommendations(test_song_idx, df_songs, similarity_df, n=5)
    print(recs.to_string(index=False))

    print("\n--- PERSONALIZED RECOMMENDATIONS ---")
    liked_songs = [0, 5, 23]

    print("User liked these songs:")
    for idx in liked_songs:
        s = df_songs.loc[idx]
        print(f"  '{s['title']}' by {s['artist']} [{s['genre']}]")

    user_profile = build_user_profile(liked_songs, feature_matrix)
    print(f"\nUser taste profile (average features):")
    print(user_profile.round(3).to_string())

    personalized = get_personalized_recommendations(
        user_profile, df_songs, feature_matrix, liked_songs, n=10
    )
    print(f"\nTop 10 personalized recommendations:")
    print(personalized.to_string(index=False))

    print("\n--- EXPLAINABILITY ---")
    for rec_idx in recs.index[:2]:
        explain_recommendation(test_song_idx, rec_idx, df_songs, feature_matrix)