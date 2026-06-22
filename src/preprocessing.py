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
    """Legge l'intestazione del CSV originale e corregge dinamicamente una colonna malformata.

    Il dataset pubblico di Steam presenta spesso un difetto di formattazione:
    esiste un'intestazione chiamata `DiscountDLC count`, ma nelle righe sottostanti i valori
    corrispondono in realtà a due campi separati separati da virgola (`Discount` e `DLC count`).
    Questa funzione intercetta l'anomalia e sdoppia l'intestazione, garantendo che le colonne
    successive non risultino sfalsate durante la lettura in Pandas.
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
    """Carica in memoria il dataset grezzo ottimizzando l'uso della RAM.

    Sfrutta `build_corrected_header` per mappare correttamente le colonne,
    ma utilizza il parametro `usecols` di Pandas per caricare esclusivamente
    le colonne definite in `RAW_COLUMNS`. Questo evita di saturare la memoria
    con decine di colonne inutili (es. descrizioni testuali lunghe o link HTML)
    che non servono alla nostra pipeline di machine learning.
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
    """Calcola il numero di lingue supportate convertendo stringhe complesse.

    Nel CSV originale, `Supported languages` è salvato come una rappresentazione testuale
    di una lista Python (es. `['English', 'Italian', 'French']`).
    La funzione:
    1. Prova a eseguire il parsing sicuro tramite `ast.literal_eval`.
    2. Se il formato è corrotto o non è una lista valida, applica un fallback
       dividendo la stringa tramite le virgole.
    Restituisce un intero che quantifica lo sforzo di localizzazione del publisher.
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
    """Estrae il genere principale per ridurre la dimensionalità del dataset.

    Steam permette di associare multipli generi (es. "Action, RPG, Indie").
    Trattare ogni combinazione come una categoria a sé stante creerebbe un'esplosione
    combinatoria (troppe variabili dummy). Per semplificare l'analisi e mantenere
    le tabelle di probabilità bayesiane compatte, estraiamo solo il primo genere elencato,
    considerandolo come quello "primario" o trainante.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return value.split(",")[0].strip()


def has_multiplayer(value: object) -> int:
    """Ingegnerizza una feature binaria (0/1) per la presenza del multiplayer.

    Analizza il campo testuale `Categories` alla ricerca di parole chiave specifiche.
    La ricerca della sottostringa "co-op".
    Restituisce 1 se il gioco ha una componente multigiocatore, altrimenti 0.
    È una metrica fondamentale per valutare la viralità e la longevità commerciale del titolo.
    """
    if not isinstance(value, str):
        return 0
    categories = value.lower()

    multiplayer_markers = [
        "multi-player",
        "multiplayer",
        "co-op",
        "pvp",
    ]
    return int(any(marker in categories for marker in multiplayer_markers))


def derive_project_features(df: pd.DataFrame) -> pd.DataFrame:
    """Centralizza la logica di Feature Engineering.

    Trasforma le colonne grezze nelle feature di business finali:
    - Calcola `Review_Count` come somma di recensioni positive e negative (volume di interazione).
    - Calcola `Review_Score_Pct` (da 0 a 100) per quantificare la qualità percepita.
    - Converte i minuti di gioco (`playtime forever`) in ore (`Playtime_Hours`).
      Se la media è nulla o corrotta, fa un fallback sulla mediana.
    - Applica i parser per estrarre Lingue, Genere e Multiplayer.
    Ritorna un DataFrame contenente solo le colonne strettamente necessarie (`PROJECT_COLUMNS`).
    """
    derived = df.copy()

    numeric_columns = [
        "Price",
        "Positive",
        "Negative",
        "Average playtime forever",
        "Median playtime forever",
    ]
    for column in numeric_columns:
        derived[column] = pd.to_numeric(derived[column], errors="coerce").fillna(0)

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

    derived["Playtime_Hours"] = (
        derived["Average playtime forever"]
        .where(derived["Average playtime forever"] > 0, derived["Median playtime forever"])
        .div(60)
    )

    return derived[PROJECT_COLUMNS].copy()


def clean_and_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Applica i filtri di qualità dei dati e campiona il dataset.

    Logica di pulizia:
    1. Rimuove duplicati esatti basati sull'AppID.
    2. Rimuove righe senza Nome o Genere (dati non azionabili).
    3. `MIN_REVIEW_COUNT`: Elimina i giochi fantasma. Un gioco con 2 recensioni
       non ha un segnale statistico utile per addestrare un modello.
    4. `Price.between(0, 100)`: Rimuove i "troll games" venduti a prezzi irrealistici (es. 999$).
    5. Estrae un campione casuale (se > SAMPLE_SIZE) usando `RANDOM_STATE` per garantire
       che gli esperimenti documentati nella relazione siano replicabili 1:1.
    """
    cleaned = df.drop_duplicates(subset=["AppID"]).copy()
    cleaned = cleaned.dropna(subset=["Name", "Primary_Genre"])

    cleaned = cleaned[cleaned["Review_Count"] >= MIN_REVIEW_COUNT]
    cleaned = cleaned[cleaned["Price"].between(0, 100)]
    cleaned = cleaned[cleaned["Playtime_Hours"] > 0]
    cleaned = cleaned.reset_index(drop=True)

    if len(cleaned) > SAMPLE_SIZE:
        cleaned = cleaned.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    return cleaned.reset_index(drop=True)


def normalize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scala le variabili numeriche continue nell'intervallo standard [0, 1].

    Perché è fondamentale?
    Algoritmi basati sul calcolo delle distanze (come KMeans per il clustering o SVM
    per la classificazione supervisionata) sono altamente sensibili alla scala.
    Senza normalizzazione, una variabile come `Review_Count` (es. 50.000) schiaccerebbe
    completamente l'influenza del `Price` (es. 15), falsando i cluster.
    Utilizza MinMaxScaler di sklearn, con fallback su calcolo manuale.
    """
    normalized = df.copy()

    if MinMaxScaler is not None:
        scaler = MinMaxScaler()
        normalized[NUMERIC_FEATURES] = scaler.fit_transform(normalized[NUMERIC_FEATURES])
        return normalized

    # Fallback manuale: formula (x - min) / (max - min)
    for column in NUMERIC_FEATURES:
        minimum = normalized[column].min()
        maximum = normalized[column].max()
        if maximum == minimum:
            normalized[column] = 0
        else:
            normalized[column] = (normalized[column] - minimum) / (maximum - minimum)

    return normalized


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converte le variabili continue in categorie discrete (binning).

    Requisito della Rete Bayesiana:
    Pgmpy richiede che i nodi siano variabili discrete. Non può gestire il prezzo come 14.99,
    ma necessita di stati come "0" (Economico), "1" (Medio), "2" (Costoso).

    Logica:
    - Usa KBinsDiscretizer con strategia "quantile": i bin vengono creati in modo che
      ciascuno contenga circa lo stesso numero di giochi, gestendo meglio le distribuzioni sbilanciate.
    - Se sklearn non è presente, usa il fallback `pd.qcut` di Pandas.
    - Le feature categoriche (es. Genere) vengono convertite in codici numerici interi (cat.codes).
    """
    discretized = df.copy()

    for column, bins in DISCRETIZATION_BINS.items():
        if column not in discretized.columns:
            continue

        unique_values = discretized[column].nunique(dropna=True)
        if unique_values < 2:
            discretized[column] = 0
            continue

        n_bins = min(bins, unique_values)

        if KBinsDiscretizer is not None:
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
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
    """Funzione orchestratrice (Entry Point) del file di preprocessing.

    Esegue in sequenza architetturale l'intera estrazione, trasformazione e caricamento (ETL):
    1. Importa i dati crudi correggendo l'header.
    2. Deriva le logiche e metriche di progetto (Feature Engineering).
    3. Pulisce errori, outlier e seleziona il campione riproducibile (Data Cleaning).
    4. Clona e normalizza il dataset per la matematica spaziale (Machine Learning & Clustering).
    5. Clona e discretizza il dataset in stati interi probabilistici (Rete Bayesiana).
    6. Scrive fisicamente i 3 dataset su disco.
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
    print(f"Pipeline ETL completata con successo.")
    print(f"Dimensioni del Dataset Originale/Pulito: {clean.shape}")
    print(f"Dimensioni del Dataset Normalizzato (MinMax): {normalized.shape}")
    print(f"Dimensioni del Dataset Discretizzato (Quantili): {discretized.shape}")
