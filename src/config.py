"""Configurazione centrale della pipeline.

Questo modulo agisce come "Single Source of Truth" (Unica Fonte di Verità) per l'intero progetto.
Centralizzare i percorsi dei file, i parametri di machine learning e le soglie di business
permette di modificare il comportamento dell'intera pipeline senza dover toccare il codice sorgente
dei singoli script operativi.
"""

# Dataset originale e output del preprocessing
RAW_DATA_PATH = "data/raw/games.csv"  # Dataset crudo scaricato da Steam
CLEAN_DATA_PATH = "data/processed/steam_games_clean.csv"  # Dati filtrati e puliti, mantenendo la scala reale
PROCESSED_DATA_PATH = "data/processed/steam_games_normalized.csv"  # Dati scalati [0, 1] per algoritmi spaziali (KMeans, SVM)
DISCRETIZED_DATA_PATH = "data/processed/steam_games_discretized.csv"  # Dati trasformati in categorie intere per Rete Bayesiana

# Dataset arricchiti con le etichette assegnate dalla fase di clustering
CLUSTERED_CLEAN_DATA_PATH = "data/processed/steam_games_clean_clustered.csv"
CLUSTERED_NORMALIZED_DATA_PATH = "data/processed/steam_games_normalized_clustered.csv"
CLUSTERED_DISCRETIZED_DATA_PATH = "data/processed/steam_games_discretized_clustered.csv"

# File generati per documentare gli esperimenti nella relazione finale
ELBOW_RESULTS_PATH = "results/kmeans_elbow.csv"
ELBOW_PLOT_PATH = "results/kmeans_elbow.png"
CLUSTER_SUMMARY_PATH = "results/cluster_summary.csv"  # Identikit dei cluster (medie e mode)
CLASS_DISTRIBUTION_PATH = "results/class_distribution.csv"  # Analisi dello sbilanciamento delle classi target
SUPERVISED_METRICS_PATH = "results/supervised_metrics.csv"  # Performance dei modelli (F1, Accuracy, ecc.)
SUPERVISED_BEST_PARAMS_PATH = "results/supervised_best_params.csv" # Configurazione ottimale eletta da GridSearchCV
SUPERVISED_MODEL_PATH = "models/best_supervised_model.joblib"  # Miglior modello supervisionato salvato per la KB
BAYESIAN_EDGES_PATH = "results/bayesian_edges.csv"  # Archi (relazioni di influenza) appresi dalla rete
BAYESIAN_QUERIES_PATH = "results/bayesian_queries.csv"  # Risultati dell'inferenza probabilistica
CATEGORY_MAPPINGS_PATH = "results/category_mappings.csv"  # Mappa tra codici interi e categorie testuali

# File Prolog: regole manuali, fatti generati e verdetti finali.
PROLOG_RULES_PATH = "kb/publisher_rules.pl"  # Base di conoscenza statica: le logiche decisionali del publisher
PROLOG_FACTS_PATH = "kb/generated_facts.pl"  # Fatti dinamici (giochi, metriche) scritti da Python per Prolog
PROLOG_DECISIONS_PATH = "results/prolog_decisions.csv"  # Verdetti finali di approvazione/rifiuto restituiti da Prolog

# Parametri globali condivisi tra gli script.
TARGET_CLUSTER_COLUMN = "Cluster_Label"  # Nome standardizzato della variabile dipendente (Target)
RANDOM_STATE = 42  # Seed fissato per garantire la riproducibilità matematica in ogni modulo (Campionamento, SMOTE, ML)
SAMPLE_SIZE = 5000  # Dimensione massima del campione per mantenere i tempi di computazione gestibili
MIN_REVIEW_COUNT = 20  # Soglia minima di recensioni per considerare un gioco statisticamente rilevante (scarta il rumore)

# Parametri del clustering KMeans.
KMEANS_K_RANGE = range(1, 11)  # Range di K esplorato dall'Elbow Method (da 1 a 10 cluster)
SELECTED_K = 3  # Valore ottimale di K scelto visivamente dal data scientist dopo aver analizzato la curva a gomito
KMEANS_N_INIT = 10  # Numero di inizializzazioni casuali per mitigare l'effetto dei minimi locali sub-ottimali
KMEANS_MAX_ITER = 300  # Numero massimo di iterazioni concesse per far convergere i centroidi

# Parametri della valutazione supervisionata e della rete bayesiana.
CV_SPLITS = 5  # Numero di fold per la Cross-Validation (i dati vengono divisi in 5 parti)
CV_REPEATS = 3  # Quante volte ripetere la Cross-Validation per consolidare la deviazione standard
SUPERVISED_N_JOBS = 1  # Esegue GridSearchCV in modo sequenziale per evitare saturazione di RAM/CPU sui laptop
SUPERVISED_PRE_DISPATCH = "1*n_jobs"  # Limita il numero di job preparati in anticipo da joblib
BAYESIAN_MAX_INDEGREE = 3  # Limite massimo di genitori per nodo nel DAG per evitare Tabelle di Probabilità intrattabili
BAYESIAN_MEDIUM_RISK_THRESHOLD = 0.30  # Probabilità minima di recensioni basse per classificare rischio medio
BAYESIAN_HIGH_RISK_THRESHOLD = 0.45  # Probabilità minima di recensioni basse per classificare rischio alto

# Questi valori vengono letti da Python, trascritti in kb/generated_facts.pl e
# infine interrogati dalle regole di publisher_rules.pl per le decisioni di finanziamento.
MIN_GLOBAL_LANGUAGES = 3  # Requisito minimo di localizzazione per considerare il gioco pubblicabile
MIN_PREMIUM_LANGUAGES = 5  # Requisito minimo di localizzazione per considerare il supporto premium
HIGH_PRICE_THRESHOLD = 30  # Un gioco in vendita a >= $30 viene considerato un investimento "ad alto rischio"
LOW_PRICE_THRESHOLD = 10   # Un gioco in vendita a <= $10 è considerato di fascia economica

# Colonne lette dal CSV grezzo in fase di importazione per risparmiare memoria RAM
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

# Schema finale del dataset pulito dopo la fase di Feature Engineering
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

# Feature numeriche continue che richiedono di essere scalate [0,1] per SVM/KMeans
NUMERIC_FEATURES = [
    "Price",
    "Review_Score_Pct",
    "Review_Count",
    "Playtime_Hours",
    "Languages_Count",
]

# Sottoinsieme di feature impiegate esclusivamente per calcolare le distanze in KMeans.
# Non includiamo attributi categorici per mantenere lo spazio puramente geometrico.
CLUSTERING_FEATURES = [
    "Price",
    "Review_Score_Pct",
    "Review_Count",
    "Playtime_Hours",
]

# Feature pre-lancio in input alla pipeline di Machine Learning per predire il target (Cluster_Label).
# Non includono recensioni osservate, cosi il modello puo essere usato su proposte non ancora pubblicate.
SUPERVISED_FEATURES = [
    "Primary_Genre",
    "Price",
    "Playtime_Hours",
    "Languages_Count",
    "Multiplayer",
]

# Feature categoriche (es. testuali o booleane) che necessitano di trasformazione per l'ML
CATEGORICAL_FEATURES = [
    "Primary_Genre",
    "Multiplayer",
]

# Topologia completa di input alla Rete Bayesiana (le variabili osservate + il nodo target)
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

# Regole di discretizzazione quantilica per la Rete Bayesiana.
# Determina in quanti stati discreti (bin) "spezzare" una variabile continua.
# Esempio -> Price: 3 significa che i prezzi saranno mappati sugli stati 0 (Basso), 1 (Medio), 2 (Alto).
DISCRETIZATION_BINS = {
    "Price": 3,
    "Review_Score_Pct": 3,
    "Review_Count": 3,
    "Playtime_Hours": 3,
    "Languages_Count": 3,
}
