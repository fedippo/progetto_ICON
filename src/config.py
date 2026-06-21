"""Configurazione centrale della pipeline IndieLaunch Pad."""

# Percorsi dei dataset originali e dei dataset prodotti dal preprocessing.
RAW_DATA_PATH = "data/raw/games.csv"
CLEAN_DATA_PATH = "data/processed/steam_games_clean.csv"
PROCESSED_DATA_PATH = "data/processed/steam_games_normalized.csv"
DISCRETIZED_DATA_PATH = "data/processed/steam_games_discretized.csv"

# Percorsi dei dataset dopo l'aggiunta delle etichette di clustering.
CLUSTERED_CLEAN_DATA_PATH = "data/processed/steam_games_clean_clustered.csv"
CLUSTERED_NORMALIZED_DATA_PATH = "data/processed/steam_games_normalized_clustered.csv"
CLUSTERED_DISCRETIZED_DATA_PATH = "data/processed/steam_games_discretized_clustered.csv"

# Output sperimentali usati nella relazione.
ELBOW_RESULTS_PATH = "results/kmeans_elbow.csv"
ELBOW_PLOT_PATH = "results/kmeans_elbow.png"
CLUSTER_SUMMARY_PATH = "results/cluster_summary.csv"
CLASS_DISTRIBUTION_PATH = "results/class_distribution.csv"
SUPERVISED_METRICS_PATH = "results/supervised_metrics.csv"
SUPERVISED_BEST_PARAMS_PATH = "results/supervised_best_params.csv"
BAYESIAN_EDGES_PATH = "results/bayesian_edges.csv"
BAYESIAN_QUERIES_PATH = "results/bayesian_queries.csv"

# File Prolog: regole manuali, fatti generati e verdetti finali.
PROLOG_RULES_PATH = "kb/publisher_rules.pl"
PROLOG_FACTS_PATH = "kb/generated_facts.pl"
PROLOG_DECISIONS_PATH = "results/prolog_decisions.csv"

# Parametri globali condivisi tra gli script.
TARGET_CLUSTER_COLUMN = "Cluster_Label"
RANDOM_STATE = 42
SAMPLE_SIZE = 5000
MIN_REVIEW_COUNT = 20

# Parametri del clustering KMeans.
KMEANS_K_RANGE = range(1, 11)
SELECTED_K = 3
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300

# Parametri della valutazione supervisionata e della rete bayesiana.
CV_SPLITS = 5
CV_REPEATS = 3
BAYESIAN_MAX_INDEGREE = 3

# Soglie di business usate o documentate nella componente Prolog.
MAX_GRANT_BUDGET = 150000
MIN_GLOBAL_LANGUAGES = 3
MIN_PREMIUM_LANGUAGES = 5

# Colonne lette dal CSV originale. Sono solo quelle necessarie al progetto.
RAW_COLUMNS = [
    "AppID",
    "Name",
    "Price",
    "Supported languages",
    "Positive",
    "Negative",
    "Average playtime forever",
    "Median playtime forever",
    "Categories",
    "Genres",
]

# Colonne finali del dataset pulito, usate nelle fasi successive.
PROJECT_COLUMNS = [
    "AppID",
    "Name",
    "Primary_Genre",
    "Price",
    "Review_Score_Pct",
    "Review_Count",
    "Playtime_Hours",
    "Languages_Count",
    "Multiplayer",
]

# Feature numeriche da normalizzare e/o discretizzare.
NUMERIC_FEATURES = [
    "Price",
    "Review_Score_Pct",
    "Review_Count",
    "Playtime_Hours",
    "Languages_Count",
]

# Feature impiegate per il KMeans.
CLUSTERING_FEATURES = [
    "Price",
    "Review_Score_Pct",
    "Review_Count",
    "Playtime_Hours",
]

# Feature usate dai modelli supervisionati per predire Cluster_Label.
SUPERVISED_FEATURES = [
    "Primary_Genre",
    "Price",
    "Review_Score_Pct",
    "Review_Count",
    "Playtime_Hours",
    "Languages_Count",
    "Multiplayer",
]

# Feature categoriche da codificare per rete bayesiana o modelli supervisionati.
CATEGORICAL_FEATURES = [
    "Primary_Genre",
    "Multiplayer",
]

# Variabili incluse nella rete bayesiana.
BAYESIAN_FEATURES = [
    "Primary_Genre",
    "Price",
    "Review_Score_Pct",
    "Review_Count",
    "Playtime_Hours",
    "Languages_Count",
    "Multiplayer",
    "Cluster_Label",
]

# Numero di bin per discretizzare le variabili continue.
DISCRETIZATION_BINS = {
    "Price": 3,
    "Review_Score_Pct": 3,
    "Review_Count": 3,
    "Playtime_Hours": 3,
    "Languages_Count": 3,
}
