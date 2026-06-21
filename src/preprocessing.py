"""Pipeline di preprocessing per il dataset Steam Games."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import pandas as pd

try:
    from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
except ModuleNotFoundError:
    # Fallback utile quando lo script viene solo ispezionato o l'ambiente non ha sklearn.
    KBinsDiscretizer = None
    MinMaxScaler = None

from config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    DISCRETIZATION_BINS,
    DISCRETIZED_DATA_PATH,
    MIN_REVIEW_COUNT,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    PROJECT_COLUMNS,
    RANDOM_STATE,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    SAMPLE_SIZE,
)


def build_corrected_header(path: str = RAW_DATA_PATH) -> list[str]:
    """Legge l'intestazione del CSV originale e corregge la colonna fusa.

    Il dataset contiene il nome colonna `DiscountDLC count`, ma nelle righe i valori
    corrispondono a due campi separati: `Discount` e `DLC count`. Questa funzione
    costruisce quindi una nuova lista di nomi colonna allineata ai valori reali.
    """
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        original_header = next(csv.reader(csv_file))

    corrected_header: list[str] = []
    for column in original_header:
        if column == "DiscountDLC count":
            # Nel CSV i valori sono separati, ma il nome colonna e stato fuso.
            corrected_header.extend(["Discount", "DLC count"])
        else:
            corrected_header.append(column)

    return corrected_header


def load_dataset(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carica il dataset grezzo limitandosi alle colonne utili al progetto.

    Usa l'header corretto da `build_corrected_header`, cosi le colonne dopo
    `DiscountDLC count` non risultano sfalsate. Restituisce un DataFrame ancora
    non pulito, da cui saranno poi derivate le feature del progetto.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}.")

    corrected_header = build_corrected_header(path)
    return pd.read_csv(
        data_path,
        header=0,
        names=corrected_header,
        usecols=RAW_COLUMNS,
        low_memory=False,
    )


def parse_language_count(value: object) -> int:
    """Calcola quante lingue sono supportate da un gioco.

    Nel CSV il campo `Supported languages` e spesso una lista salvata come stringa,
    ad esempio `['English', 'Italian']`. La funzione prova a interpretarla come
    lista Python; se il formato non e valido, usa una separazione semplice su virgola.
    """
    if not isinstance(value, str) or value.strip() in {"", "[]"}:
        return 0

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Se il valore non e una lista Python valida, uso una separazione semplice.
        return len([item for item in value.split(",") if item.strip()])

    if isinstance(parsed, list):
        return len(parsed)
    return 0


def extract_primary_genre(value: object) -> str | None:
    """Estrae il genere principale da una lista di generi Steam.

    Steam puo associare piu generi allo stesso gioco, separati da virgole. Per
    mantenere una feature categorica semplice e difendibile nella relazione, viene
    usato il primo genere come `Primary_Genre`.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return value.split(",")[0].strip()


def has_multiplayer(value: object) -> int:
    """Restituisce 1 se le categorie Steam indicano funzionalita multiplayer.

    La funzione cerca parole chiave come `multi-player`, `co-op` o `pvp` dentro il
    campo `Categories`. Il risultato e una variabile binaria usata nei modelli e
    nella rete bayesiana.
    """
    if not isinstance(value, str):
        return 0
    categories = value.lower()
    # Steam usa nomi diversi per modalita multiplayer, co-op e PvP.
    multiplayer_markers = [
        "multi-player",
        "multiplayer",
        "co-op",
        "pvp",
        "online co-op",
        "lan co-op",
    ]
    return int(any(marker in categories for marker in multiplayer_markers))


def derive_project_features(df: pd.DataFrame) -> pd.DataFrame:
    """Costruisce le feature finali a partire dalle colonne grezze.

    Questa funzione centralizza il feature engineering: converte i numeri,
    calcola volume e percentuale delle recensioni, estrae genere principale,
    conta le lingue, deriva il multiplayer e converte il playtime in ore.
    Restituisce solo le colonne elencate in `PROJECT_COLUMNS`.
    """
    derived = df.copy()

    # Converto le colonne numeriche prima di calcolare feature derivate.
    numeric_columns = [
        "Price",
        "Positive",
        "Negative",
        "Average playtime forever",
        "Median playtime forever",
    ]
    for column in numeric_columns:
        derived[column] = pd.to_numeric(derived[column], errors="coerce").fillna(0)

    # Review_Count misura il volume di attenzione, Review_Score_Pct la ricezione media.
    review_count = derived["Positive"] + derived["Negative"]
    derived["Review_Count"] = review_count
    derived["Review_Score_Pct"] = (
        derived["Positive"].div(review_count).fillna(0).mul(100)
    )
    derived["Primary_Genre"] = derived["Genres"].apply(extract_primary_genre)
    derived["Languages_Count"] = derived["Supported languages"].apply(
        parse_language_count
    )
    derived["Multiplayer"] = derived["Categories"].apply(has_multiplayer)

    # Se il playtime medio e nullo, uso la mediana come alternativa; poi converto in ore.
    derived["Playtime_Hours"] = (
        derived["Average playtime forever"]
        .where(derived["Average playtime forever"] > 0, derived["Median playtime forever"])
        .div(60)
    )

    return derived[PROJECT_COLUMNS].copy()


def clean_and_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Pulisce il dataset e applica un campionamento riproducibile.

    Rimuove duplicati, giochi senza nome/genere, record con poche recensioni e
    prezzi estremi. Se dopo i filtri restano piu di `SAMPLE_SIZE` giochi, estrae un
    campione casuale con seed fisso, cosi gli esperimenti sono ripetibili.
    """
    cleaned = df.drop_duplicates(subset=["AppID"]).copy()
    cleaned = cleaned.dropna(subset=["Name", "Primary_Genre"])

    # Il filtro sulle recensioni evita giochi senza segnale commerciale sufficiente.
    cleaned = cleaned[cleaned["Review_Count"] >= MIN_REVIEW_COUNT]
    cleaned = cleaned[cleaned["Price"].between(0, 100)]
    cleaned = cleaned[cleaned["Playtime_Hours"] >= 0]
    cleaned = cleaned.reset_index(drop=True)

    if len(cleaned) > SAMPLE_SIZE:
        cleaned = cleaned.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    return cleaned.reset_index(drop=True)


def normalize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza le feature numeriche nell'intervallo [0, 1].

    La normalizzazione e necessaria per KMeans e SVM, perche lavorano con distanze
    o margini influenzati dalla scala. Se `scikit-learn` e disponibile usa
    `MinMaxScaler`; altrimenti applica manualmente la formula Min-Max.
    """
    normalized = df.copy()

    if MinMaxScaler is not None:
        scaler = MinMaxScaler()
        normalized[NUMERIC_FEATURES] = scaler.fit_transform(normalized[NUMERIC_FEATURES])
        return normalized

    # Fallback manuale equivalente alla normalizzazione Min-Max.
    for column in NUMERIC_FEATURES:
        minimum = normalized[column].min()
        maximum = normalized[column].max()
        if maximum == minimum:
            normalized[column] = 0
        else:
            normalized[column] = (normalized[column] - minimum) / (maximum - minimum)

    return normalized


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trasforma feature continue e categoriche in stati discreti.

    La rete bayesiana richiede variabili discrete gestibili. Le feature numeriche
    vengono divise in bin, mentre le categoriche vengono convertite in codici
    interi. Il dataset risultante viene usato da `bayesian_network.py`.
    """
    discretized = df.copy()

    for column, bins in DISCRETIZATION_BINS.items():
        if column not in discretized.columns:
            continue

        unique_values = discretized[column].nunique(dropna=True)
        if unique_values < 2:
            # Se una feature e costante, la rete bayesiana vede un unico stato.
            discretized[column] = 0
            continue

        n_bins = min(bins, unique_values)

        if KBinsDiscretizer is not None:
            # Strategia quantile: i bin tendono ad avere numerosita simile.
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
            )
            discretized[column] = discretizer.fit_transform(
                discretized[[column]]
            ).astype(int)
        else:
            # Fallback pandas quando sklearn non e disponibile.
            discretized[column] = pd.qcut(
                discretized[column],
                q=n_bins,
                labels=False,
                duplicates="drop",
            ).fillna(0).astype(int)

    for column in CATEGORICAL_FEATURES:
        if column in discretized.columns:
            # La rete bayesiana lavora meglio con stati discreti numerici.
            discretized[column] = discretized[column].astype("category").cat.codes

    return discretized


def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Esegue l'intera fase di preprocessing e salva i tre dataset intermedi.

    L'ordine e: caricamento CSV, feature engineering, pulizia/campionamento,
    normalizzazione e discretizzazione. Restituisce i tre DataFrame anche al
    chiamante, oltre a salvarli nella cartella `data/processed`.
    """
    raw_df = load_dataset()
    project_df = derive_project_features(raw_df)
    clean_df = clean_and_sample(project_df)
    normalized_df = normalize_numeric_features(clean_df)
    discretized_df = discretize_features(clean_df)

    Path(CLEAN_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    normalized_df.to_csv(PROCESSED_DATA_PATH, index=False)
    discretized_df.to_csv(DISCRETIZED_DATA_PATH, index=False)

    return clean_df, normalized_df, discretized_df


if __name__ == "__main__":
    clean, normalized, discretized = run_preprocessing()
    print(f"Clean dataset shape: {clean.shape}")
    print(f"Normalized dataset shape: {normalized.shape}")
    print(f"Discretized dataset shape: {discretized.shape}")
