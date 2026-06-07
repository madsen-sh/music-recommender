 Music Recommendation System
A content-based music recommendation engine built from scratch in Python.
Suggests songs based on audio feature similarity and learns from a user's
listening history to build a personal taste profile.

 What It Does

Recommends songs similar to any given track using cosine similarity
Builds a user taste profile from liked songs and generates personalized picks
Boosts recommendations using genre preference and popularity scoring
Explains why each song was recommended (most similar/different features)
Visualizes the entire song library in 2D space using PCA


 ML Concepts Used
ConceptWhere It's UsedCosine SimilarityMeasuring how similar two songs are in feature spacePCA (Dimensionality Reduction)Compressing 8 features → 2D for visualizationFeature Normalization (Min-Max)Putting all features on equal 0–1 scaleUser Profile VectorAveraging liked song features to represent tasteScore BlendingCombining similarity + popularity + genre into one scoreExplainabilityShowing which features drove each recommendation

 Project Structure
music_recommender/
│
├── data.py          # Generates 500 songs with realistic audio features
├── features.py      # Normalizes and engineers feature matrix
├── recommender.py   # Cosine similarity engine + user profile + ranking
├── visualize.py     # 3 research-quality charts
├── main.py          # Runs the full pipeline end to end
└── outputs/         # Generated charts saved here

 Visualizations
Song Similarity Map
Every song plotted in 2D space using PCA. Songs that sound alike
cluster together. Colors represent genres.
Audio Feature Radar
Compares average audio features across all 8 genres. Shows clearly
how Electronic differs from Classical, or Hip-Hop from Jazz.
Genre Distribution
Compares genre counts in the full library vs in the recommendations —
confirms the recommender correctly surfaces preferred genres.

 How To Run
1. Install dependencies:
pip install numpy pandas matplotlib scikit-learn scipy
2. Run the full pipeline:
python main.py
Charts will be saved to the outputs/ folder automatically.

 How It Works
Step 1 — Data
500 songs are generated with 8 audio features each:
tempo, energy, danceability, valence, acousticness,
instrumentalness, loudness, duration. Genre-based adjustments
make the data realistic — Rock songs are more energetic, Classical
songs are more acoustic.
Step 2 — Feature Engineering
All features are normalized to a 0–1 scale using Min-Max scaling.
This is critical — without it, tempo (60–200 BPM) would dominate
every similarity calculation just because its raw numbers are bigger.
Step 3 — Similarity Engine
A 500×500 cosine similarity matrix is built — every song compared
to every other song. Each cell contains a score from -1 (opposite)
to 1 (identical).
Step 4 — User Profile
When a user likes multiple songs, their feature vectors are averaged
into a single taste profile vector. New recommendations are then
ranked by how close they are to this profile — not just to one song.
Step 5 — Smart Ranking
Final score = similarity score + popularity boost + genre preference boost.
This mirrors how real recommendation systems blend multiple signals.

 Key Design Decisions
Why cosine similarity and not Euclidean distance?
Cosine similarity measures the angle between feature vectors, not
the raw distance. This means a quiet acoustic song and a loud acoustic
song can still be recognized as similar in style — the magnitude
difference doesn't matter, only the direction.
Why Min-Max over Standard scaling?
Audio features like energy and danceability have natural 0–1 boundaries.
Min-Max respects those boundaries. Standard scaling (z-score) would
push values outside that range, which loses the interpretability.
Why PCA for visualization?
8-dimensional data is impossible to plot directly. PCA finds the 2
directions that capture the most variance in the data and projects
everything onto those axes — so the 2D chart still reflects the true
structure of the feature space.

 Built With

Python 3.14
NumPy — numerical computing
Pandas — data manipulation
scikit-learn — MinMaxScaler, cosine similarity, PCA
Matplotlib — visualizations