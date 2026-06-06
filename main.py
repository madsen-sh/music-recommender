from data import generate_songs
from features import engineer_features
from recommender import (build_similarity_matrix, get_recommendations,
                         build_user_profile, get_personalized_recommendations, 
                         explain_recommendation)
from visualize import (plot_similarity_map, plot_feature_radar,
                         plot_genre_distribution) 


def main():
    print("=" *50)
    print("Music RECOMMENDATION SYSTEM")
    print("=" *50)

    print("\n[1/4] Generating song dataset...")
    df_songs = generate_songs(n_songs = 500)
    print(f"   {len(df_songs)} songs created across {df_songs['genre'].nunique()} genres.")

    print("\n[2/4] Engineering features...")
    df_engineered, feature_matrix = engineer_features(df_songs)
    print(f"  Feature matrix: {feature_matrix.shape}")

    print("\n[3/4] Building similarity matrix...")
    similarity_df = build_similarity_matrix(feature_matrix)
    print(f" Matrix shape: {similarity_df.shape}")

    print("\n[4/4] Generating visualizations...")
    liked_songs = [0, 5, 23]
    user_profile = build_user_profile(liked_songs, feature_matrix)
    recommendations = get_personalized_recommendations(
        user_profile, df_songs, feature_matrix, liked_songs, n = 10
    )

    plot_similarity_map(df_songs, feature_matrix)
    plot_feature_radar(df_songs, liked_songs)
    plot_genre_distribution(df_songs, recommendations)

    print("\n" + "=" * 50)
    print(" SINGLE SONG RECOMMENDATIONS")
    print("=" * 50)

    seed = df_songs.loc[0]
    print("\nSeed: {seed['title']} by {seed['artist']} [{seed['genre']}]")
    recs = get_recommendations(0, df_songs, similarity_df, n = 5)
    print(recs.to_string(index = False))

    print("\n" + "=" * 50)
    print(" PERSONALIZED RECOMMENDATIONS")
    print("=" * 50)
    print("\nBased on these liked songs:")

    for idx in liked_songs:
        s = df_songs.loc[idx]
        print(f"  '{s['title']}' by {s['artist']} [{s['genre']}]")

    print(f"\nTop 10 picks for you:")
    print(recommendations.to_string(index = False))

    print("\n" + "=" *50)
    print(" EXPLAINABILITY")
    print("=" * 50)
    for rec_idx in recs.index[:3]:
        explain_recommendation(0, rec_idx, df_songs, feature_matrix)

    print("\nDone. Charts saved to outputs/")

if __name__ == "__main__":
    main()