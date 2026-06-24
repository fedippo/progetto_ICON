"""Integrazione Python-Prolog per le decisioni di finanziamento del publisher.

Questo modulo funge da strato di governance finale:
1. Converte i dati del dataset in fatti Prolog (`generate_prolog_facts`).
2. Interroga la Knowledge Base (kb/publisher_rules.pl) tramite l'interfaccia `pyswip`.
3. Estrae i verdetti decisionali (Approva/Revisione/Rifiuta) per ogni gioco.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import load

from bayesian_network import estimate_bayesian_risks
from config import (
    CLUSTERED_CLEAN_DATA_PATH,
    HIGH_PRICE_THRESHOLD,
    LOW_PRICE_THRESHOLD,
    MIN_GLOBAL_LANGUAGES,
    MIN_PREMIUM_LANGUAGES,
    PROLOG_DECISIONS_PATH,
    PROLOG_FACTS_PATH,
    PROLOG_RULES_PATH,
    SUPERVISED_FEATURES,
    SUPERVISED_MODEL_PATH,
)


DEMO_GAMES = [
    # Caso di successo assicurato: Strategia longevo, prezzo equo, super localizzato, rischio minimo.
    {
        "name": "stellar_strategy_master",
        "genre": "strategy",
        "primary_genre": "Strategy",
        "price": 19.99,
        "hours": 50,
        "languages": 12,
        "multiplayer": 0,
    },
    # Caso intermedio: gioco coerente e ben localizzato, ma con prezzo alto.
    {
        "name": "short_puzzle_deluxe",
        "genre": "puzzle",
        "primary_genre": "Puzzle",
        "price": 34.99,
        "hours": 4,
        "languages": 6,
        "multiplayer": 0,
    },
    # Caso respinto: caratteristiche buone, ma localizzazione insufficiente.
    {
        "name": "underlocalized_action",
        "genre": "action",
        "primary_genre": "Action",
        "price": 19.99,
        "hours": 9,
        "languages": 1,
        "multiplayer": 1,
    },
    # Caso respinto: RPG troppo corto, prezzo alto e multiplayer.
    {
        "name": "overpriced_multiplayer_rpg",
        "genre": "rpg",
        "primary_genre": "RPG",
        "price": 49.99,
        "hours": 8,
        "languages": 5,
        "multiplayer": 1,
    },
]


def load_supervised_model():
    """Carica il miglior modello supervisionato salvato dalla fase di training."""
    model_path = Path(SUPERVISED_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Supervised model not found at {model_path}. Run src/supervised_learning.py first."
        )
    return load(model_path)


def build_model_input(games: list[dict]) -> pd.DataFrame:
    """Costruisce il DataFrame di input per il classificatore pre-lancio.

    I giochi demo sono descritti con valori reali leggibili da Prolog. Il modello,
    pero, e stato addestrato sui dataset normalizzati: per questo le variabili
    numeriche vengono riscalate usando minimi e massimi del dataset pulito.
    """
    clean_path = Path(CLUSTERED_CLEAN_DATA_PATH)
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean clustered dataset not found at {clean_path}. Run src/clustering.py first."
        )

    clean_df = pd.read_csv(clean_path)
    rows = []
    for game in games:
        rows.append(
            {
                "Primary_Genre": game["primary_genre"],
                "Price": game["price"],
                "Playtime_Hours": game["hours"],
                "Languages_Count": game["languages"],
                "Multiplayer": game["multiplayer"],
            }
        )

    model_input = pd.DataFrame(rows)
    for column in ["Price", "Playtime_Hours", "Languages_Count"]:
        minimum = clean_df[column].min()
        maximum = clean_df[column].max()
        if maximum == minimum:
            model_input[column] = 0
        else:
            model_input[column] = (model_input[column] - minimum) / (maximum - minimum)

    return model_input[SUPERVISED_FEATURES]


def predict_commercial_profiles(games: list[dict]) -> dict[str, str]:
    """Predice il cluster commerciale dei giochi demo con il modello supervisionato."""
    model = load_supervised_model()
    model_input = build_model_input(games)
    predictions = model.predict(model_input)
    return {
        game["name"]: f"cluster_{int(prediction)}"
        for game, prediction in zip(games, predictions)
    }


def generate_prolog_facts(games: list[dict] = DEMO_GAMES) -> Path:
    """Genera il file di fatti dinamici utilizzato dalla Knowledge Base Prolog.

    Ruolo nella pipeline:
    Le regole decisionali del publisher sono definite staticamente nel file
    'publisher_rules.pl', mentre i dati dei giochi cambiano ad ogni esecuzione.
    Questa funzione realizza il collegamento tra Python e Prolog trasformando
    giochi da valutare in fatti logici. I casi dimostrativi contengono caratteristiche
    disponibili o stimabili prima della pubblicazione: genere, prezzo, durata prevista,
    lingue e multiplayer. Il cluster commerciale viene predetto dal
    miglior modello supervisionato salvato, mentre il verdetto viene derivato da Prolog.

    Funzionamento:
    1. Crea la cartella di destinazione se non esiste.
    2. Scrive nel file le soglie di business definite in config.py
       (lingue e prezzo).
    3. Converte ogni gioco nel predicato:
       gioco(Nome, Genere, Prezzo, Ore, Lingue, Multiplayer).
    4. Predice il cluster con il miglior modello supervisionato e lo scrive come:
       predizione_commerciale(Gioco, Cluster).
    5. Stima il rischio di recensioni basse con la rete bayesiana e lo scrive come:
       rischio_bayesiano(Gioco, Rischio).
    6. Salva il file .pl risultante e restituisce il percorso generato.

    Il file prodotto costituisce la base di fatti interrogata successivamente
    dal motore SWI-Prolog.
    """
    facts_path = Path(PROLOG_FACTS_PATH)
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    predicted_clusters = predict_commercial_profiles(games)
    bayesian_risks = estimate_bayesian_risks(games, predicted_clusters)

    lines = [
        "% File generato da src/prolog_reasoning.py.",
        "",
        f"soglia_lingue_globale({MIN_GLOBAL_LANGUAGES}).",
        f"soglia_lingue_premium({MIN_PREMIUM_LANGUAGES}).",
        f"soglia_prezzo_alto({HIGH_PRICE_THRESHOLD}).",
        f"soglia_prezzo_basso({LOW_PRICE_THRESHOLD}).",
        "",
    ]

    # Ogni fatto contiene solo informazioni pre-pubblicazione.
    for game in games:
        lines.append(
            "gioco("
            f"{game['name']}, "
            f"{game['genre']}, "
            f"{game['price']}, "
            f"{game['hours']}, "
            f"{game['languages']}, "
            f"{game['multiplayer']}"
            ")."
        )

    lines.append("")
    for game in games:
        lines.append(
            f"predizione_commerciale({game['name']}, {predicted_clusters[game['name']]})."
        )

    lines.append("")
    for game in games:
        risk_info = bayesian_risks[game["name"]]
        lines.append(f"rischio_bayesiano({game['name']}, {risk_info['risk']}).")

    lines.append("")
    for game in games:
        risk_info = bayesian_risks[game["name"]]
        probability = float(risk_info["low_review_probability"])
        lines.append(
            f"probabilita_recensioni_basse({game['name']}, {probability:.6f})."
        )

    facts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return facts_path


def import_pyswip():
    """Importa dinamicamente l'interfaccia Python-SWI Prolog.

    La libreria PySwip richiede sia il pacchetto Python 'pyswip'
    sia una installazione funzionante di SWI-Prolog.

    L'import viene eseguito solo quando necessario per evitare che
    il semplice caricamento del modulo fallisca in ambienti dove
    Prolog non è installato.

    Restituisce:
        Classe Prolog fornita da PySwip.

    Solleva:
        ModuleNotFoundError se PySwip o SWI-Prolog non sono disponibili.
    """
    try:
        from pyswip import Prolog
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Missing pyswip dependency. Install requirements.txt and SWI-Prolog before running this script."
        ) from exc
    return Prolog


def consult_knowledge_base():
    """Inizializza il motore Prolog e carica la Knowledge Base completa.

    Funzionamento:
    1. Importa dinamicamente la classe Prolog tramite PySwip.
    2. Crea una nuova istanza del motore logico SWI-Prolog.
    3. Verifica l'esistenza dei file:
       - publisher_rules.pl (regole statiche)
       - generated_facts.pl (fatti dinamici)
    4. Carica entrambi i file tramite il predicato 'consult'.
    5. Restituisce il motore Prolog pronto per essere interrogato.

    L'ordine di caricamento è importante:
    prima vengono definite le regole generali del publisher,
    poi i fatti specifici della sessione corrente.
    """
    Prolog = import_pyswip()
    prolog = Prolog()

    rules_path = Path(PROLOG_RULES_PATH).resolve()
    facts_path = Path(PROLOG_FACTS_PATH).resolve()

    if not rules_path.exists():
        raise FileNotFoundError(f"Missing Prolog rules file: {rules_path}")
    if not facts_path.exists():
        raise FileNotFoundError(f"Missing Prolog facts file: {facts_path}")

    # Prima le regole stabili del publisher, poi i fatti generati.
    prolog.consult(str(rules_path))
    prolog.consult(str(facts_path))
    return prolog


def query_single_value(prolog, query: str, variable: str) -> str | None:
    """Esegue una query Prolog e restituisce il primo valore trovato.

    Parametri:
        prolog: istanza del motore SWI-Prolog.
        query: query logica da eseguire.
        variable: nome della variabile da estrarre.

    Funzionamento:
    - Esegue la query tramite PySwip.
    - Converte il generatore dei risultati in una lista.
    - Se non esistono soluzioni restituisce None.
    - In caso contrario estrae il valore della variabile richiesta
      dalla prima soluzione disponibile.

    È utile quando la query è progettata per produrre un unico risultato,
    come il verdetto finale associato a un gioco.
    """
    results = list(prolog.query(query))
    if not results:
        return None
    return str(results[0][variable])


def query_all_values(prolog, query: str, variable: str) -> list[str]:
    """Esegue una query Prolog e raccoglie tutte le soluzioni trovate.

    A differenza di 'query_single_value', questa funzione non si ferma
    alla prima soluzione ma scorre completamente il generatore prodotto
    da PySwip.

    Per ogni soluzione:
    - estrae il valore associato alla variabile richiesta;
    - converte il risultato in stringa;
    - lo inserisce nella lista finale.

    Viene utilizzata quando un predicato può produrre più risposte,
    ad esempio l'elenco completo delle violazioni bloccanti associate
    ad un progetto.
    """
    return [str(row[variable]) for row in prolog.query(query)]


def unique_preserving_order(values: list[str]) -> list[str]:
    """Rimuove i duplicati preservando l'ordine originale.

    Le query Prolog possono generare la stessa soluzione attraverso
    percorsi logici differenti. In questi casi è possibile ottenere
    valori ripetuti.

    La funzione:
    1. Mantiene un insieme ('seen') dei valori già incontrati.
    2. Scorre la lista nell'ordine originale.
    3. Inserisce nella lista finale solo gli elementi non ancora visti.

    Restituisce quindi una sequenza di valori unici senza alterare
    l'ordine con cui sono stati generati dal motore Prolog.
    """
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def query_derived_profile(prolog, game: str) -> str:
    """Restituisce il profilo commerciale derivato dalle regole Prolog."""
    profile_checks = [
        ("successo", f"profilo_successo({game})"),
        ("intermedio", f"profilo_intermedio({game})"),
        ("rischio", f"profilo_rischio({game})"),
        ("medio", f"rischio_medio({game})"),
    ]
    for label, query in profile_checks:
        if list(prolog.query(query)):
            return label
    return "non_derivato"


def build_decision_rows(prolog) -> list[dict]:
    """Costruisce la tabella finale dei risultati decisionali.

    Rappresenta il punto di integrazione tra il motore logico Prolog
    e il layer analitico Python basato su Pandas.

    Funzionamento:
    1. Recupera tutti i giochi presenti nella Knowledge Base
       interrogando il predicato 'gioco/6'.
    2. Per ciascun gioco esegue una serie di interrogazioni:
       - verdetto(Game, Verdetto)
       - approva_finanziamento(Game)
       - richiede_revisione(Game)
       - rifiuta_finanziamento(Game)
       - violazione_bloccante(Game, Motivo)
    3. Elimina eventuali duplicati nelle violazioni.
    4. Costruisce un dizionario contenente:
       - nome del gioco;
       - verdetto finale;
       - flag booleani per approvazione, revisione o rifiuto;
       - elenco delle violazioni rilevate.
    5. Inserisce il risultato nella struttura finale.

    Il valore restituito è una lista di record pronta per essere
    convertita in DataFrame Pandas e successivamente esportata in CSV.
    """
    game_names = query_all_values(prolog, "gioco(Nome, _, _, _, _, _)", "Nome")
    rows = []

    for game in game_names:
        # Verdetto principale: approvato, revisione o rifiutato.
        verdict = query_single_value(prolog, f"verdetto({game}, Verdetto)", "Verdetto")
        predicted_cluster = query_single_value(
            prolog,
            f"predizione_commerciale({game}, Cluster)",
            "Cluster",
        )
        bayesian_risk = query_single_value(
            prolog,
            f"rischio_bayesiano({game}, Rischio)",
            "Rischio",
        )
        low_review_probability = query_single_value(
            prolog,
            f"probabilita_recensioni_basse({game}, Probabilita)",
            "Probabilita",
        )
        derived_profile = query_derived_profile(prolog, game)

        # Le violazioni spiegano perche un gioco viene bloccato.
        violations = query_all_values(
            prolog,
            f"violazione_bloccante({game}, Motivo)",
            "Motivo",
        )
        violations = unique_preserving_order(violations)

        rows.append(
            {
                "Game": game,
                "Predicted_Cluster": predicted_cluster or "nessun_cluster",
                "Bayesian_Risk": bayesian_risk or "nessun_rischio",
                "Low_Review_Probability": low_review_probability or "n/a",
                "Derived_Profile": derived_profile,
                "Verdict": verdict or "nessun_verdetto",
                "Approved": bool(list(prolog.query(f"approva_finanziamento({game})"))),
                "Needs_Review": bool(list(prolog.query(f"richiede_revisione({game})"))),
                "Rejected": bool(list(prolog.query(f"rifiuta_finanziamento({game})"))),
                "Blocking_Violations": ";".join(violations),
            }
        )

    return rows


def run_prolog_reasoning() -> pd.DataFrame:
    """Esegue l'intera pipeline di ragionamento simbolico.

    Questa funzione rappresenta l'entry point del modulo Prolog e
    coordina tutte le fasi necessarie alla generazione dei verdetti.

    Sequenza operativa:
    1. Rigenera il file dei fatti Prolog a partire dai casi dimostrativi correnti.
    2. Inizializza il motore SWI-Prolog.
    3. Carica regole e fatti nella Knowledge Base.
    4. Interroga la base di conoscenza per ottenere le decisioni.
    5. Converte i risultati in un DataFrame Pandas.
    6. Salva il report finale nel file:
       results/prolog_decisions.csv
    7. Restituisce il DataFrame per ulteriori analisi o visualizzazioni.

    Il risultato finale rappresenta il verdetto di business ottenuto
    combinando:
    - regole esperte codificate manualmente;
    - profilo commerciale e rischio derivati da Prolog;
    - caratteristiche pre-pubblicazione rappresentate come fatti logici.
    """
    generate_prolog_facts()
    prolog = consult_knowledge_base()
    decisions_df = pd.DataFrame(build_decision_rows(prolog))
    Path(PROLOG_DECISIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    decisions_df.to_csv(PROLOG_DECISIONS_PATH, index=False)
    return decisions_df


if __name__ == "__main__":
    decisions = run_prolog_reasoning()
    print(decisions.to_string(index=False))
