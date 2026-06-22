"""Clustering KMeans ed Elbow Method per i profili commerciali Steam."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    # Gestione fallback: il CSV con i risultati numerici dell'Elbow Method
    # viene comunque calcolato e salvato anche se l'ambiente non ha matplotlib per i grafici.
    plt = None

try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError:
    # Se scikit-learn non è installato nell'ambiente locale, la variabile viene impostata a None.
    # Il codice gestirà questo scenario implementando una logica di blocco o fallback sicuro.
    KMeans = None

from config import (
    CLEAN_DATA_PATH,
    CLUSTER_SUMMARY_PATH,
    CLUSTERED_CLEAN_DATA_PATH,
    CLUSTERED_DISCRETIZED_DATA_PATH,
    CLUSTERED_NORMALIZED_DATA_PATH,
    CLUSTERING_FEATURES,
    DISCRETIZED_DATA_PATH,
    ELBOW_PLOT_PATH,
    ELBOW_RESULTS_PATH,
    KMEANS_K_RANGE,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    SELECTED_K,
    TARGET_CLUSTER_COLUMN,
)


def load_processed_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carica i tre dataset paralleli generati nella fase precedente di preprocessing.

    Il funzionamento si basa sul caricamento coordinato di tre varianti dello stesso dataset:
    1. Dataset 'Clean': Contiene i dati originali non scalati. Sarà fondamentale alla fine
       del clustering per calcolare le medie reali e interpretare il significato di business dei cluster.
    2. Dataset 'Normalized': Dati scalati tra 0 e 1. Viene usato direttamente dall'algoritmo KMeans
       per evitare che variabili con scale grandi (es. numero di recensioni) dominino il calcolo della distanza.
    3. Dataset 'Discretized': Dati suddivisi in bin energetici/numerici. Riceverà le stesse etichette
       di cluster per consentire alla successiva Rete Bayesiana di lavorare su variabili totalmente discrete.
    """
    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    normalized_df = pd.read_csv(PROCESSED_DATA_PATH)
    discretized_df = pd.read_csv(DISCRETIZED_DATA_PATH)
    return clean_df, normalized_df, discretized_df


def validate_clustering_features(df: pd.DataFrame) -> None:
    """Verifica formale della presenza di tutte le feature richieste per il clustering.

    Prima di avviare algoritmi computazionalmente intensivi, questa funzione analizza le colonne
    del DataFrame normalizzato confrontandole con l'elenco 'CLUSTERING_FEATURES'.
    Se una o più colonne sono assenti, interrompe l'esecuzione lanciando un KeyError dettagliato,
    prevenendo fallimenti opachi o comportamenti imprevisti di scikit-learn.
    """
    missing = [column for column in CLUSTERING_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing clustering features: {missing}")


def compute_inertia(data: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Calcola matematicamente l'Inerzia, ovvero la WCSS (Within-Cluster Sum of Squares).

    Funzionamento matematico:
    - Riceve la matrice dei dati, l'array delle etichette assegnate a ciascun punto e la matrice dei centroidi.
    - Sfrutta il broadcasting di NumPy per calcolare lo scostamento: `data - centers[labels]`.
      Per ogni punto viene sottratto il vettore del rispettivo centroide.
    - Moltiplica l'array per se stesso elemento per elemento (equivalente al quadrato della distanza euclidea)
      e ne esegue la somma totale (`np.sum`).
    - Il valore ottenuto misura la compattezza geometrica dei cluster: più è basso, più i punti sono vicini ai loro centri.
    """
    distances = data - centers[labels]
    return float(np.sum(distances * distances))


def fallback_kmeans(
    data: np.ndarray,
    n_clusters: int,
    n_init: int = KMEANS_N_INIT,
    max_iter: int = KMEANS_MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Algoritmo KMeans nativo sviluppato in NumPy come soluzione di fallback.

    Viene eseguito se scikit-learn non è installato nell'ambiente. La logica interna prevede:
    1. Controllo di consistenza: verifica che il numero di cluster non superi il numero di record.
    2. Ciclo di inizializzazione (`n_init` volte): per evitare di convergere in minimi locali deboli,
       l'algoritmo viene riavviato più volte estraendo ogni volta `n_clusters` punti casuali come centroidi iniziali.
    3. Fase di ottimizzazione (fino a `max_iter` iterazioni):
       - Calcola la distanza euclidea tra ogni punto e tutti i centroidi sfruttando l'espansione dimensionale:
         `data[:, None, :] - centers[None, :, :]`.
       - Assegna ogni punto al centroide più vicino (`argmin(axis=1)`).
       - Aggiorna la posizione dei centroidi calcolando la media geometrica di tutti i punti assegnati a quel cluster.
       - Se la posizione dei centroidi non cambia rispetto al ciclo precedente (`np.allclose`), la convergenza è raggiunta.
    4. Selezione del modello migliore: conserva la combinazione di etichette e centroidi che minimizza l'inerzia globale.
    """
    if n_clusters > len(data):
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot be greater than "
            f"the number of data points ({len(data)})."
        )
    best_labels = None
    best_centers = None
    best_inertia = float("inf")
    rng = np.random.default_rng(random_state)

    for _ in range(n_init):
        # Inizializzazione casuale dei centroidi, ripetuta per ridurre soluzioni deboli.
        center_indexes = rng.choice(len(data), size=n_clusters, replace=False)
        centers = data[center_indexes].copy()

        for _ in range(max_iter):
            # Assegna ogni gioco al centroide piu vicino.
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for cluster_id in range(n_clusters):
                members = data[labels == cluster_id]
                if len(members) > 0:
                    # Il nuovo centroide e la media dei punti assegnati al cluster.
                    new_centers[cluster_id] = members.mean(axis=0)

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        inertia = compute_inertia(data, labels, centers)
        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

    return best_labels, best_centers, best_inertia


def run_kmeans(data: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Interfaccia unificata (Wrapper) per l'esecuzione del KMeans.

    Controlla dinamicamente la presenza di scikit-learn:
    - Se disponibile, delega il calcolo alla classe standard `KMeans` di sklearn.
    - Se assente, reindirizza la chiamata sulla funzione locale `fallback_kmeans`.
    In questo modo, il resto del codice interagisce con un'API identica, che restituisce sempre:
    (array_etichette, matrice_centroidi, valore_inerzia).
    """
    if KMeans is not None:
        model = KMeans(
            n_clusters=n_clusters,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(data)
        return labels, model.cluster_centers_, float(model.inertia_)

    return fallback_kmeans(data, n_clusters=n_clusters)


def run_elbow_method(data: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Calcola l'andamento dell'Inerzia su diversi valori di K per l'Elbow Method.

    Funzionamento:
    - Cicla all'interno del range predefinito (es. da 1 a 10 cluster).
    - Chiama la funzione `run_kmeans` per ogni K, registrando il rispettivo valore di inerzia.
    - Intercetta l'iterazione in cui `k == SELECTED_K` e ne memorizza le etichette di assegnazione.
    Ciò consente di ricavare le label ottimali direttamente durante lo screening del grafico a gomito,
    evitando di dover riaddestrare il modello finale in un secondo momento.
    """
    rows = []
    selected_labels = None
    for k in KMEANS_K_RANGE:
        labels, _, inertia = run_kmeans(data, n_clusters=k)
        rows.append({"k": k, "inertia": inertia})
        if k == SELECTED_K:
            selected_labels = labels
    return pd.DataFrame(rows), selected_labels


def save_elbow_plot(elbow_df: pd.DataFrame) -> None:
    """Genera e memorizza il grafico dell'andamento dell'Inerzia (Curva a Gomito).

    Il grafico permette di individuare visivamente il "punto di gomito" (elbow), ovvero il valore di K
    oltre il quale l'aggiunta di ulteriori cluster non riduce in modo significativo la varianza interna.
    Se 'matplotlib' non è installato, la funzione interrompe l'esecuzione in modo sicuro,
    poiché i dati numerici sono comunque già stati salvati in formato CSV.
    """
    if plt is None:
        return

    Path(ELBOW_PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
    ax.set_title("KMeans Elbow Method")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(KMEANS_K_RANGE))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ELBOW_PLOT_PATH, dpi=160)
    plt.close(fig)


def attach_cluster_labels(
    clean_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    discretized_df: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sincronizza le etichette dei cluster assegnate su tutti i dataset della pipeline.

    Per garantire la coerenza atomica dei dati, questa funzione inserisce l'array numerico
    delle label come nuova colonna `Cluster_Label` (definita in `TARGET_CLUSTER_COLUMN`)
    in tutti e tre i DataFrame (clean, normalizzato, discretizzato). Questo garantisce che la riga
    i-esima di ogni file faccia sempre riferimento allo stesso identico cluster nelle fasi successive.
    """
    clean_clustered = clean_df.copy()
    normalized_clustered = normalized_df.copy()
    discretized_clustered = discretized_df.copy()

    clean_clustered[TARGET_CLUSTER_COLUMN] = labels
    normalized_clustered[TARGET_CLUSTER_COLUMN] = labels
    discretized_clustered[TARGET_CLUSTER_COLUMN] = labels

    return clean_clustered, normalized_clustered, discretized_clustered


def build_cluster_summary(clean_clustered: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabella di sintesi statistica per delineare i profili commerciali dei cluster.

    Logica di funzionamento:
    - Raggruppa il dataset 'Clean' (scala reale originaria) in base alla colonna del cluster.
    - Calcola metriche mirate per ciascun gruppo: conteggio dei giochi, prezzo medio,
      punteggio recensioni medio, mediana del volume di recensioni, ore di gioco medie, lingue medie e tasso di multiplayer.
    - Estrae il genere più frequente (`Top_Genre`) applicando la moda statistica (`mode()`) su ogni gruppo.
    Questa tabella trasforma gli ID numerici astratti di KMeans in "profili aziendali" interpretabili dal management.
    """
    summary = (
        clean_clustered.groupby(TARGET_CLUSTER_COLUMN)
        .agg(
            Games=("AppID", "count"),
            Avg_Price=("Price", "mean"),
            Avg_Review_Score=("Review_Score_Pct", "mean"),
            Median_Reviews=("Review_Count", "median"),
            Avg_Playtime_Hours=("Playtime_Hours", "mean"),
            Avg_Languages=("Languages_Count", "mean"),
            Multiplayer_Rate=("Multiplayer", "mean"),
            Top_Genre=("Primary_Genre", lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown"),
        )
        .reset_index()
    )
    return summary.round(3)


def run_clustering() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestratore principale (Entry Point) del modulo di clustering.

    Esegue in sequenza logica rigida i seguenti passi:
    1. Carica i tre dataset intermedi (clean, normalized, discretized).
    2. Valida la conformità delle colonne.
    3. Isola le feature numeriche normalizzate e le converte in una matrice NumPy per l'elaborazione geometrica.
    4. Esegue l'Elbow Method estraendo l'andamento delle inerzie e le label per il `SELECTED_K`.
    5. Assegna le etichette e genera i grafici/tabelle riassuntive.
    6. Scrive tutti gli output finali (compresi i 3 dataset integrati con il cluster) nelle rispettive cartelle.
    """
    clean_df, normalized_df, discretized_df = load_processed_datasets()
    validate_clustering_features(normalized_df)

    # Il clustering usa dati normalizzati per evitare dominanza delle feature piu grandi.
    data = normalized_df[CLUSTERING_FEATURES].to_numpy(dtype=float)
    elbow_df, labels = run_elbow_method(data)
    if labels is None:
        raise ValueError(
            f"SELECTED_K ({SELECTED_K}) non è in KMEANS_K_RANGE. "
            "Aggiorna la configurazione in config.py."
        )

    clean_clustered, normalized_clustered, discretized_clustered = attach_cluster_labels(
        clean_df, normalized_df, discretized_df, labels
    )
    summary = build_cluster_summary(clean_clustered)

    Path(ELBOW_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    elbow_df.to_csv(ELBOW_RESULTS_PATH, index=False)
    save_elbow_plot(elbow_df)
    summary.to_csv(CLUSTER_SUMMARY_PATH, index=False)

    # Esporta i 3 dataset finali "Clustered"
    clean_clustered.to_csv(CLUSTERED_CLEAN_DATA_PATH, index=False)
    normalized_clustered.to_csv(CLUSTERED_NORMALIZED_DATA_PATH, index=False)
    discretized_clustered.to_csv(CLUSTERED_DISCRETIZED_DATA_PATH, index=False)

    return elbow_df, summary


if __name__ == "__main__":
    elbow, cluster_summary = run_clustering()
    print("Elbow results")
    print(elbow.to_string(index=False))
    print("\nCluster summary")
    print(cluster_summary.to_string(index=False))
