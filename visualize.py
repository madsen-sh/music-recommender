import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

def plot_similarity_map(df_songs, feature_matrix):

    pca = PCA(n_components = 2)

    coords    = pca.fit_transform(feature_matrix.values)
    explained = pca.explained_variance_ratio_

    genres = df_songs["genre"].values
    unique_genres = sorted(set(genres))
    genre_to_int = {g: i for i, g in enumerate(unique_genres)}
    color_indices = [genre_to_int[g] for g in genres]

    fig, ax = plt.subplots(figsize = (12, 8))

    colors = cm.tab10(np.linspace(0, 1, len(unique_genres)))

    for i , genre in enumerate(unique_genres):
        mask = genres == genre
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c = [colors[i]],
            label = genre,
            alpha = 0.6,
            s = 40
        )

    ax.set_xlabel(f"Component 1 ({explained[0]*100:.1f}% variance)", fontsize = 11)
    ax.set_ylabel(f"Component 2 ({explained[1]*100:.1f}% variance)", fontsize = 11)
    ax.set_title("Song Similatiry Map - PCA of Audio Features", fontsize = 14, fontweight ="bold")
    ax.legend(title="Genre", bbox_to_anchor = (1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig("outputs/01_similarity_map.png", dpi=150 , bbox_inches = "tight")
    plt.close()
    print("  Saved: outputs/01_similarity_map.png")

def plot_feature_radar(df_songs, liked_indices):

    genres = df_songs["genre"].unique()
    feature_cols = ["energy", "danceability", "valence",
                    "acousticness", "instrumentalness"]

    genre_means = df_songs.groupby("genre")[feature_cols].mean()

    num_features = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = cm.tab10(np.linspace(0, 1, len(genre_means)))

    for i, (genre, row) in enumerate(genre_means.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=genre, color=colors[i])
        ax.fill(angles, values, alpha=0.05, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Average Audio Features by Genre", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    plt.savefig("outputs/02_feature_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/02_feature_radar.png")


def plot_genre_distribution(df_songs,recommendations):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    genre_counts = df_songs["genre"].value_counts()
    axes[0].bar(genre_counts.index, genre_counts.values,
                color = cm.tab10(np.linspace(0, 1, len(genre_counts)))),
    axes[0].set_title("Song Count by Genre", fontsize = 13, fontweight = "bold")
    axes[0].set_xlabel("Genre")
    axes[0].set_ylabel("Number of Songs")
    axes[0].tick_params(axis = "x", rotation = 30)

    rec_genres = recommendations["genre"].value_counts()
    axes[1].bar(rec_genres.index, rec_genres.values,
                color = cm.tab10(np.linspace(0, 1, len(rec_genres))))
    axes[1].set_title("Genre Distributin in Recommendation", fontsize = 13, fontweight = "bold")
    axes[1].set_xlabel("Genre")
    axes[1].set_ylabel("Number of Recommendation")
    axes[1].tick_params(axis = "x", rotation = 30)

    plt.suptitle("Music Recommender - Genre Analysis", fontsize = 15,
                 fontweight = "bold", y = 1.02)
    plt.tight_layout()
    plt.savefig("outputs/03_genre_distribution.png", dpi = 150 , bbox_inches= "tight")
    plt.close()
    print(" Saved: outputs/03_genre_distribution.png")



if __name__ == "__main__":
    from data import generate_songs
    from features import engineer_features
    from recommender import build_similarity_matrix, build_user_profile
    from recommender import get_personalized_recommendations

    print("Building data...")
    df_songs = generate_songs()
    df_engineered, feature_matrix = engineer_features(df_songs)
    similarity_df = build_similarity_matrix(feature_matrix)

    liked_songs = [0, 5, 23]
    user_profile = build_user_profile(liked_songs, feature_matrix)
    recommendations = get_personalized_recommendations(
        user_profile, df_songs, feature_matrix, liked_songs, n=10
    )

    print("Generating charts...")
    plot_similarity_map(df_songs, feature_matrix)
    plot_feature_radar(df_songs, liked_songs)
    plot_genre_distribution(df_songs, recommendations)
    print("\nALL charts saved to outputs/")