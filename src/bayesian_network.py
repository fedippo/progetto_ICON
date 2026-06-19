"""Bayesian Network learning and inference for Steam market risk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    BAYESIAN_EDGES_PATH,
    BAYESIAN_FEATURES,
    BAYESIAN_MAX_INDEGREE,
    BAYESIAN_QUERIES_PATH,
    CLUSTERED_DISCRETIZED_DATA_PATH,
    RANDOM_STATE,
    TARGET_CLUSTER_COLUMN,
)


def import_pgmpy_dependencies():
    """Import pgmpy lazily so the file can be inspected without installed deps."""
    try:
        from pgmpy.estimators import BicScore, HillClimbSearch, MaximumLikelihoodEstimator
        from pgmpy.inference import VariableElimination
        try:
            from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
        except ImportError:
            from pgmpy.models import BayesianNetwork as BayesianModel
        try:
            from pgmpy.parameter_estimator import DiscreteMLE
        except ImportError:
            DiscreteMLE = None
    except ImportError:
        try:
            from pgmpy.estimators import BIC as BicScore
            from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
            from pgmpy.inference import VariableElimination
            try:
                from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
            except ImportError:
                from pgmpy.models import BayesianNetwork as BayesianModel
            try:
                from pgmpy.parameter_estimator import DiscreteMLE
            except ImportError:
                DiscreteMLE = None
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Missing pgmpy dependency. Install requirements.txt before running this script."
            ) from exc

    return {
        "BayesianModel": BayesianModel,
        "BicScore": BicScore,
        "HillClimbSearch": HillClimbSearch,
        "MaximumLikelihoodEstimator": MaximumLikelihoodEstimator,
        "DiscreteMLE": DiscreteMLE,
        "VariableElimination": VariableElimination,
    }


def load_bayesian_dataset() -> pd.DataFrame:
    """Load discretized and clustered data for Bayesian Network learning."""
    data_path = Path(CLUSTERED_DISCRETIZED_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Discretized clustered dataset not found at {data_path}. Run clustering.py first."
        )

    df = pd.read_csv(data_path)
    missing = [column for column in BAYESIAN_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing Bayesian Network columns: {missing}")

    bayesian_df = df[BAYESIAN_FEATURES].copy()
    for column in bayesian_df.columns:
        bayesian_df[column] = bayesian_df[column].astype(int)

    return bayesian_df


def learn_structure(df: pd.DataFrame, pgmpy):
    """Learn the DAG structure with HillClimbSearch and BIC."""
    search = pgmpy["HillClimbSearch"](df)
    try:
        return search.estimate(
            scoring_method=pgmpy["BicScore"](df),
            max_indegree=BAYESIAN_MAX_INDEGREE,
        )
    except TypeError:
        return search.estimate(
            scoring_method="bic-d",
            max_indegree=BAYESIAN_MAX_INDEGREE,
        )


def fit_model(df: pd.DataFrame, structure, pgmpy):
    """Fit CPDs for the learned network."""
    model = pgmpy["BayesianModel"](structure.edges())
    if pgmpy["DiscreteMLE"] is not None:
        model.fit(df, estimator=pgmpy["DiscreteMLE"]())
    else:
        model.fit(df, estimator=pgmpy["MaximumLikelihoodEstimator"])
    model.check_model()
    return model


def save_edges(model) -> pd.DataFrame:
    """Save learned edges for documentation and inspection."""
    edges_df = pd.DataFrame(list(model.edges()), columns=["Source", "Target"])
    Path(BAYESIAN_EDGES_PATH).parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(BAYESIAN_EDGES_PATH, index=False)
    return edges_df


def most_common_state(df: pd.DataFrame, column: str) -> int:
    """Return the most frequent discrete state for a column."""
    return int(df[column].mode().iloc[0])


def highest_state(df: pd.DataFrame, column: str) -> int:
    """Return the highest observed discrete state for a column."""
    return int(df[column].max())


def lowest_state(df: pd.DataFrame, column: str) -> int:
    """Return the lowest observed discrete state for a column."""
    return int(df[column].min())


def query_cluster_distribution(inference, evidence: dict) -> list[dict]:
    """Query P(Cluster_Label | evidence)."""
    result = inference.query(
        variables=[TARGET_CLUSTER_COLUMN],
        evidence=evidence,
        show_progress=False,
    )

    rows = []
    for state, probability in enumerate(result.values):
        rows.append(
            {
                "Query": f"P({TARGET_CLUSTER_COLUMN} | {evidence})",
                "Variable": TARGET_CLUSTER_COLUMN,
                "State": state,
                "Probability": float(probability),
            }
        )
    return rows


def query_low_review_risk(inference, evidence: dict) -> list[dict]:
    """Query P(Review_Score_Pct | evidence), used as a reception-risk proxy."""
    result = inference.query(
        variables=["Review_Score_Pct"],
        evidence=evidence,
        show_progress=False,
    )

    rows = []
    for state, probability in enumerate(result.values):
        rows.append(
            {
                "Query": f"P(Review_Score_Pct | {evidence})",
                "Variable": "Review_Score_Pct",
                "State": state,
                "Probability": float(probability),
            }
        )
    return rows


def run_business_queries(df: pd.DataFrame, model, pgmpy) -> pd.DataFrame:
    """Run a compact set of business-oriented inference queries."""
    inference = pgmpy["VariableElimination"](model)

    common_genre = most_common_state(df, "Primary_Genre")
    high_price = highest_state(df, "Price")
    low_price = lowest_state(df, "Price")
    long_playtime = highest_state(df, "Playtime_Hours")
    multiplayer = 1 if 1 in set(df["Multiplayer"]) else most_common_state(df, "Multiplayer")

    query_rows = []
    query_rows.extend(
        query_cluster_distribution(
            inference,
            evidence={"Primary_Genre": common_genre, "Price": high_price},
        )
    )
    query_rows.extend(
        query_cluster_distribution(
            inference,
            evidence={"Price": low_price, "Playtime_Hours": long_playtime},
        )
    )
    query_rows.extend(
        query_low_review_risk(
            inference,
            evidence={"Price": high_price, "Multiplayer": multiplayer},
        )
    )

    queries_df = pd.DataFrame(query_rows)
    queries_df.to_csv(BAYESIAN_QUERIES_PATH, index=False)
    return queries_df


def run_bayesian_network() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn the Bayesian Network and run inference queries."""
    pgmpy = import_pgmpy_dependencies()
    df = load_bayesian_dataset()
    structure = learn_structure(df, pgmpy)
    model = fit_model(df, structure, pgmpy)
    edges_df = save_edges(model)
    queries_df = run_business_queries(df, model, pgmpy)
    return edges_df, queries_df


if __name__ == "__main__":
    edges, queries = run_bayesian_network()
    print("Learned edges")
    print(edges.to_string(index=False))
    print("\nInference queries")
    print(queries.to_string(index=False))
