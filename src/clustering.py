"""KMeans clustering and Elbow Method for Steam market profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError:
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
    """Load clean, normalized and discretized datasets."""
    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    normalized_df = pd.read_csv(PROCESSED_DATA_PATH)
    discretized_df = pd.read_csv(DISCRETIZED_DATA_PATH)
    return clean_df, normalized_df, discretized_df


def validate_clustering_features(df: pd.DataFrame) -> None:
    """Check that all clustering features are available."""
    missing = [column for column in CLUSTERING_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing clustering features: {missing}")


def compute_inertia(data: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Compute within-cluster sum of squares."""
    distances = data - centers[labels]
    return float(np.sum(distances * distances))


def fallback_kmeans(
    data: np.ndarray,
    n_clusters: int,
    n_init: int = KMEANS_N_INIT,
    max_iter: int = KMEANS_MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Small deterministic KMeans fallback used when scikit-learn is unavailable."""
    best_labels = None
    best_centers = None
    best_inertia = float("inf")
    rng = np.random.default_rng(random_state)

    for _ in range(n_init):
        center_indexes = rng.choice(len(data), size=n_clusters, replace=False)
        centers = data[center_indexes].copy()

        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for cluster_id in range(n_clusters):
                members = data[labels == cluster_id]
                if len(members) > 0:
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
    """Run KMeans with scikit-learn when installed, otherwise use the fallback."""
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


def run_elbow_method(data: np.ndarray) -> pd.DataFrame:
    """Compute inertia for the configured K range."""
    rows = []
    for k in KMEANS_K_RANGE:
        _, _, inertia = run_kmeans(data, n_clusters=k)
        rows.append({"k": k, "inertia": inertia})
    return pd.DataFrame(rows)


def save_elbow_plot(elbow_df: pd.DataFrame) -> None:
    """Save an Elbow Method plot when matplotlib is available."""
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
    """Add the selected cluster labels to every processed dataset."""
    clean_clustered = clean_df.copy()
    normalized_clustered = normalized_df.copy()
    discretized_clustered = discretized_df.copy()

    clean_clustered[TARGET_CLUSTER_COLUMN] = labels
    normalized_clustered[TARGET_CLUSTER_COLUMN] = labels
    discretized_clustered[TARGET_CLUSTER_COLUMN] = labels

    return clean_clustered, normalized_clustered, discretized_clustered


def build_cluster_summary(clean_clustered: pd.DataFrame) -> pd.DataFrame:
    """Summarize clusters on original-scale features for interpretation."""
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
            Top_Genre=("Primary_Genre", lambda values: values.mode().iloc[0]),
        )
        .reset_index()
    )
    return summary.round(3)


def run_clustering() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Elbow Method, selected KMeans and save clustered datasets."""
    clean_df, normalized_df, discretized_df = load_processed_datasets()
    validate_clustering_features(normalized_df)

    data = normalized_df[CLUSTERING_FEATURES].to_numpy(dtype=float)
    elbow_df = run_elbow_method(data)
    labels, _, _ = run_kmeans(data, n_clusters=SELECTED_K)

    clean_clustered, normalized_clustered, discretized_clustered = attach_cluster_labels(
        clean_df, normalized_df, discretized_df, labels
    )
    summary = build_cluster_summary(clean_clustered)

    Path(ELBOW_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    elbow_df.to_csv(ELBOW_RESULTS_PATH, index=False)
    save_elbow_plot(elbow_df)
    summary.to_csv(CLUSTER_SUMMARY_PATH, index=False)
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
