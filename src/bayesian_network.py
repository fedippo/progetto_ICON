"""Apprendimento e inferenza di una rete bayesiana per il rischio commerciale."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from config import (
    BAYESIAN_HIGH_RISK_THRESHOLD,
    BAYESIAN_EDGES_PATH,
    BAYESIAN_FEATURES,
    BAYESIAN_MAX_INDEGREE,
    BAYESIAN_MEDIUM_RISK_THRESHOLD,
    BAYESIAN_QUERIES_PATH,
    CATEGORY_MAPPINGS_PATH,
    CLUSTERED_CLEAN_DATA_PATH,
    CLUSTERED_DISCRETIZED_DATA_PATH,
    TARGET_CLUSTER_COLUMN,
)

def import_pgmpy_dependencies():
    """Importa pgmpy gestendo le differenze architetturali tra le varie versioni della libreria.

    Il modulo esegue una serie di blocchi try-except per garantire la compatibilità sia con
    le versioni più vecchie di pgmpy sia con quelle più recenti. In particolare:
    - Recupera 'BicScore' o 'BIC' per la valutazione della struttura.
    - Cerca 'DiscreteBayesianNetwork' (introdotto nelle nuove API) facendo fallback su 'BayesianModel'.
    - Risolve la posizione di 'DiscreteMLE', che è stata spostata tra i moduli 'estimators'
      e 'parameter_estimator' nel corso dello sviluppo della libreria.

    Restituisce un dizionario contenente le classi necessarie, isolando il resto dello script
    dai breaking changes di pgmpy.
    """
    try:
        import pgmpy
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Missing pgmpy dependency. Install requirements.txt before running this script."
        ) from exc

    # BicScore si chiamava BIC nelle versioni piu vecchie di pgmpy.
    try:
        from pgmpy.estimators import BicScore
    except ImportError:
        from pgmpy.estimators import BIC as BicScore

    from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination

    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
    except ImportError:
        from pgmpy.models import BayesianNetwork as BayesianModel

    # NUOVA GESTIONE IMPORT: cerca DiscreteMLE anche nel nuovo modulo parameter_estimator
    try:
        from pgmpy.parameter_estimator import DiscreteMLE
    except ImportError:
        try:
            from pgmpy.estimators import DiscreteMLE
        except ImportError:
            DiscreteMLE = None

    return {
        "BayesianModel": BayesianModel,
        "BicScore": BicScore,
        "HillClimbSearch": HillClimbSearch,
        "MaximumLikelihoodEstimator": MaximumLikelihoodEstimator,
        "DiscreteMLE": DiscreteMLE,
        "VariableElimination": VariableElimination,
    }


def load_bayesian_dataset() -> pd.DataFrame:
    """Carica e prepara il dataset per l'addestramento della rete bayesiana.

    Il funzionamento prevede:
    1. La verifica dell'esistenza del file discretizzato e clusterizzato generato dai task precedenti.
    2. Il controllo formale che tutte le colonne definite in 'BAYESIAN_FEATURES' siano presenti,
       sollevando un errore in caso contrario per evitare fallimenti silenziosi.
    3. Il casting forzato di tutte le feature a interi (int). Questo passaggio è cruciale
       perché i modelli bayesiani discreti di pgmpy richiedono che gli stati siano
       rappresentati da numeri interi (es. 0, 1, 2) e non da float o stringhe.
    """
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
        # Tutte le variabili della rete devono essere stati discreti interi.
        bayesian_df[column] = bayesian_df[column].astype(int)

    return bayesian_df


def learn_structure(df: pd.DataFrame, pgmpy):
    """Apprende automaticamente la topologia (Directed Acyclic Graph) della rete bayesiana dai dati.

    Funzionamento dell'algoritmo:
    - Utilizza 'HillClimbSearch', un algoritmo di ricerca euristica che parte da un grafo vuoto
      e iterativamente aggiunge, rimuove o inverte un arco alla volta per trovare la struttura migliore.
    - Valuta la bontà delle strutture candidate usando il 'BicScore' (Bayesian Information Criterion).
      Il BIC premia la capacità della rete di spiegare i dati (likelihood) ma penalizza
      fortemente la complessità del modello (troppi archi), prevenendo l'overfitting.
    - Impone il limite 'BAYESIAN_MAX_INDEGREE' per restringere il numero massimo di nodi "genitore"
      che un nodo "figlio" può avere. Questo mantiene le Tabelle di Probabilità Condizionata (CPD)
      piccole, calcolabili rapidamente e facilmente interpretabili in ambito business.
    """
    search = pgmpy["HillClimbSearch"](df)

    def estimate_with_compatible_api(scoring_method):
        try:
            return search.estimate(
                scoring_method=scoring_method,
                max_indegree=BAYESIAN_MAX_INDEGREE,
                show_progress=False,
            )
        except TypeError:
            return search.estimate(
                scoring_method=scoring_method,
                max_indegree=BAYESIAN_MAX_INDEGREE,
            )

    try:
        return estimate_with_compatible_api(pgmpy["BicScore"](df))
    except TypeError:
        # Compatibilita con API pgmpy che accettano il nome dello score come stringa.
        return estimate_with_compatible_api("bic-d")


def fit_model(df: pd.DataFrame, structure, pgmpy):
    """Stima le Tabelle di Probabilità Condizionata (CPD) della rete bayesiana.

    Dopo aver definito la struttura (gli archi), questa funzione calcola le probabilità.
    Utilizza la Stima di Massima Verosimiglianza (MLE - Maximum Likelihood Estimation),
    che conta semplicemente le frequenze relative degli stati nel dataset.
    Ad esempio, conta quante volte un gioco appartiene al cluster X dato che il suo prezzo è Y.
    Infine, chiama 'check_model()' per validare matematicamente la rete (assicurandosi
    che le probabilità in ogni CPD sommino esattamente a 1).
    """
    model = pgmpy["BayesianModel"](structure.edges())
    model.add_nodes_from(df.columns)

    if pgmpy["DiscreteMLE"] is not None:
        # API recente: fit richiede un estimatore discreto inizializzato (con le parentesi tonde vuote)
        model.fit(df, estimator=pgmpy["DiscreteMLE"]())
    else:
        # API precedente: fit accetta la classe MaximumLikelihoodEstimator (senza parentesi)
        model.fit(df, estimator=pgmpy["MaximumLikelihoodEstimator"])

    model.check_model()
    return model


def save_edges(model) -> pd.DataFrame:
    """Estrae ed esporta la lista degli archi (relazioni causali/condizionali) della rete.

    Converte l'oggetto 'edges()' del modello pgmpy (una lista di tuple) in un DataFrame Pandas
    con colonne "Source" e "Target", per poi salvarlo in un file CSV. Questo è utile
    per documentare quali variabili influenzano altre variabili senza dover ricorrere
    alla visualizzazione grafica del DAG.
    """
    edges_df = pd.DataFrame(list(model.edges()), columns=["Source", "Target"])
    Path(BAYESIAN_EDGES_PATH).parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(BAYESIAN_EDGES_PATH, index=False)
    return edges_df


def most_common_state(df: pd.DataFrame, column: str) -> int:
    """Identifica lo stato discreto (bin) più frequente per una data variabile.

    Usa il metodo 'mode()' di pandas per estrarre la moda statistica.
    Viene impiegato per creare scenari di inferenza "tipici" o di base (baseline).
    """
    return int(df[column].mode().iloc[0])


def highest_state(df: pd.DataFrame, column: str) -> int:
    """Identifica lo stato discreto di valore numerico massimo per una data variabile.

    Estrae il massimo (max) per automatizzare la costruzione di query probabilistiche
    su scenari limite superiori, ad esempio "prezzo molto alto".
    """
    return int(df[column].max())


def lowest_state(df: pd.DataFrame, column: str) -> int:
    """Restituisce lo stato discreto minimo osservato.

    Per le variabili discretizzate del progetto corrisponde in genere allo stato
    'basso', ad esempio prezzo basso.
    """
    return int(df[column].min())


def query_cluster_distribution(inference, evidence: dict) -> list[dict]:
    """Esegue l'inferenza esatta per calcolare la probabilità di appartenenza ai cluster.

    Il metodo utilizza l'algoritmo di Eliminazione delle Variabili ('VariableElimination')
    per calcolare la distribuzione marginale della variabile target ('Cluster_Label')
    condizionata all'evidenza fornita (un dizionario di variabili osservate, es. genere e prezzo).
    Formatta poi il risultato in una lista di dizionari facilmente convertibile in DataFrame.
    """
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
    """Esegue l'inferenza esatta per stimare il rischio di recensioni negative.

    Simile a 'query_cluster_distribution', ma cambia il target: interroga la rete sulla
    variabile 'Review_Score_Pct'. L'obiettivo di business è analizzare la probabilità
    che uno specifico scenario (es. gioco costoso e multiplayer) produca uno stato 0
    (recensioni basse), indicando così un elevato rischio commerciale.
    """
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


def train_bayesian_model() -> tuple[pd.DataFrame, object, dict]:
    """Addestra la rete e restituisce dataset, modello e dipendenze pgmpy."""
    pgmpy = import_pgmpy_dependencies()
    df = load_bayesian_dataset()
    structure = learn_structure(df, pgmpy)
    model = fit_model(df, structure, pgmpy)
    return df, model, pgmpy


def load_category_code_map() -> dict[str, dict[str, int]]:
    """Carica la mappa categoria -> codice usata dal dataset discretizzato."""
    mapping_path = Path(CATEGORY_MAPPINGS_PATH)
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Category mappings not found at {mapping_path}. Run preprocessing.py first."
        )

    mappings_df = pd.read_csv(mapping_path)
    code_map: dict[str, dict[str, int]] = {}
    for _, row in mappings_df.iterrows():
        column = str(row["Column"])
        code_map.setdefault(column, {})[str(row["Value"])] = int(row["Code"])
    return code_map


def numeric_state_from_reference(
    clean_df: pd.DataFrame,
    discretized_df: pd.DataFrame,
    column: str,
    value: float,
) -> int:
    """Assegna a un valore reale lo stato discreto coerente con il preprocessing."""
    reference = (
        pd.DataFrame(
            {
                "value": clean_df[column],
                "state": discretized_df[column].astype(int),
            }
        )
        .dropna()
        .sort_values("value")
    )

    for state in sorted(reference["state"].unique()):
        state_values = reference.loc[reference["state"] == state, "value"]
        if value <= state_values.max():
            return int(state)

    return int(reference["state"].max())


def classify_low_review_risk(probability: float) -> str:
    """Traduce la probabilita di recensioni basse in una classe leggibile da Prolog."""
    if probability >= BAYESIAN_HIGH_RISK_THRESHOLD:
        return "alto"
    if probability >= BAYESIAN_MEDIUM_RISK_THRESHOLD:
        return "medio"
    return "basso"


def build_game_evidence(
    game: dict,
    predicted_cluster: str,
    clean_df: pd.DataFrame,
    discretized_df: pd.DataFrame,
    category_codes: dict[str, dict[str, int]],
) -> dict[str, int]:
    """Converte un gioco demo in evidenza discreta per l'inferenza bayesiana."""
    genre_codes = category_codes.get("Primary_Genre", {})
    evidence = {
        "Primary_Genre": genre_codes.get(
            str(game["primary_genre"]),
            most_common_state(discretized_df, "Primary_Genre"),
        ),
        "Price": numeric_state_from_reference(
            clean_df, discretized_df, "Price", float(game["price"])
        ),
        "Playtime_Hours": numeric_state_from_reference(
            clean_df, discretized_df, "Playtime_Hours", float(game["hours"])
        ),
        "Languages_Count": numeric_state_from_reference(
            clean_df, discretized_df, "Languages_Count", float(game["languages"])
        ),
        "Multiplayer": int(game["multiplayer"]),
        TARGET_CLUSTER_COLUMN: int(str(predicted_cluster).replace("cluster_", "")),
    }
    return evidence


def estimate_bayesian_risks(
    games: list[dict],
    predicted_clusters: dict[str, str],
) -> dict[str, dict[str, float | str]]:
    """Stima il rischio bayesiano dei giochi che verranno valutati da Prolog."""
    clean_path = Path(CLUSTERED_CLEAN_DATA_PATH)
    discretized_path = Path(CLUSTERED_DISCRETIZED_DATA_PATH)
    if not clean_path.exists() or not discretized_path.exists():
        raise FileNotFoundError("Clustered datasets not found. Run clustering.py first.")

    clean_df = pd.read_csv(clean_path)
    discretized_df = pd.read_csv(discretized_path)
    _, model, pgmpy = train_bayesian_model()
    inference = pgmpy["VariableElimination"](model)
    category_codes = load_category_code_map()

    risks: dict[str, dict[str, float | str]] = {}
    for game in games:
        evidence = build_game_evidence(
            game,
            predicted_clusters[game["name"]],
            clean_df,
            discretized_df,
            category_codes,
        )
        result = inference.query(
            variables=["Review_Score_Pct"],
            evidence=evidence,
            show_progress=False,
        )
        low_review_probability = float(result.values[0])
        risks[game["name"]] = {
            "risk": classify_low_review_risk(low_review_probability),
            "low_review_probability": low_review_probability,
        }

    return risks


def run_business_queries(df: pd.DataFrame, model, pgmpy) -> pd.DataFrame:
    """Orchestra e calcola un set di scenari decisionali tramite inferenza probabilistica.

    Questo metodo:
    1. Inizializza l'algoritmo di Eliminazione delle Variabili.
    2. Usa i metodi di supporto (es. most_common_state, highest_state) per determinare
       dinamicamente i valori degli stati discreti presenti nel dataset.
    3. Formula tre casi di studio specifici per il publisher:
       - Impatto sul cluster di un genere comune venduto a prezzo alto.
       - Impatto sul cluster di un gioco economico ma con alta longevità.
       - Rischio di recensioni negative per un gioco multiplayer venduto a prezzo alto.
    4. Raccoglie tutti i risultati e li esporta in un file CSV riassuntivo.
    """
    inference = pgmpy["VariableElimination"](model)

    # Gli stati sono discreti: 0=basso, 1=medio, 2=alto per le feature discretizzate.
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
    Path(BAYESIAN_QUERIES_PATH).parent.mkdir(parents=True, exist_ok=True)
    queries_df.to_csv(BAYESIAN_QUERIES_PATH, index=False)
    return queries_df


def run_bayesian_network() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry point principale per l'intera pipeline bayesiana.

    Si occupa di coordinare in sequenza logica l'intero processo:
    1. Importazione sicura delle librerie.
    2. Caricamento del dataset.
    3. Apprendimento della struttura (archi).
    4. Addestramento dei parametri (CPD).
    5. Salvataggio della topologia su disco.
    6. Esecuzione e salvataggio delle query di business.
    Restituisce i DataFrame degli archi e delle query per log e ispezioni.
    """
    df, model, pgmpy = train_bayesian_model()
    edges_df = save_edges(model)
    queries_df = run_business_queries(df, model, pgmpy)
    return edges_df, queries_df


if __name__ == "__main__":
    edges, queries = run_bayesian_network()
    print("Learned edges")
    print(edges.to_string(index=False))
    print("\nInference queries")
    print(queries.to_string(index=False))
