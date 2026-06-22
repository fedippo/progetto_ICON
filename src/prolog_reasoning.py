"""Integrazione Python-Prolog per le decisioni di finanziamento del publisher.

Questo modulo funge da strato di governance finale:
1. Converte i dati del dataset in fatti Prolog (`generate_prolog_facts`).
2. Interroga la Knowledge Base (kb/publisher_rules.pl) tramite l'interfaccia `pyswip`.
3. Estrae i verdetti decisionali (Approva/Revisione/Rifiuta) per ogni gioco.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    GOOD_REVIEW_SCORE_THRESHOLD,
    HIGH_PRICE_THRESHOLD,
    LOW_PRICE_THRESHOLD,
    LOW_REVIEW_SCORE_THRESHOLD,
    MAX_GRANT_BUDGET,
    MIN_GLOBAL_LANGUAGES,
    MIN_PREMIUM_LANGUAGES,
    PROLOG_DECISIONS_PATH,
    PROLOG_FACTS_PATH,
    PROLOG_RULES_PATH,
    REDUCED_GRANT_BUDGET,
)


DEMO_GAMES = [
    # Caso positivo: RPG coerente, localizzato e con rischio basso.
    {
        "name": "hades_like",
        "genre": "rpg",
        "price": 24.99,
        "hours": 30,
        "languages": 8,
        "multiplayer": 0,
        "budget": 120000,
        "review_score": 88,
        "cluster": "cluster_0",
        "risk": "basso",
    },
    # Caso intermedio: review sotto soglia (72 < 75) e rischio non accettabile
    # (rischio medio con prezzo alto >= 30). Entrambe le condizioni bloccano l'approvazione;
    {
        "name": "short_puzzle_deluxe",
        "genre": "puzzle",
        "price": 34.99,
        "hours": 4,
        "languages": 6,
        "multiplayer": 0,
        "budget": 60000,
        "review_score": 72,
        "cluster": "cluster_2",
        "risk": "medio",
    },
    # Caso respinto: cluster positivo, ma localizzazione insufficiente.
    {
        "name": "underlocalized_action",
        "genre": "action",
        "price": 19.99,
        "hours": 9,
        "languages": 1,
        "multiplayer": 1,
        "budget": 90000,
        "review_score": 81,
        "cluster": "cluster_0",
        "risk": "basso",
    },
    # Caso respinto: cluster rischioso, RPG troppo corto e rischio bayesiano alto.
    {
        "name": "overpriced_multiplayer_rpg",
        "genre": "rpg",
        "price": 49.99,
        "hours": 8,
        "languages": 5,
        "multiplayer": 1,
        "budget": 180000,
        "review_score": 55,
        "cluster": "cluster_1",
        "risk": "alto",
    },
]


def generate_prolog_facts(games: list[dict] = DEMO_GAMES) -> Path:
    """Genera il file di fatti dinamici utilizzato dalla Knowledge Base Prolog.

    Ruolo nella pipeline:
    Le regole decisionali del publisher sono definite staticamente nel file
    'publisher_rules.pl', mentre i dati dei giochi cambiano ad ogni esecuzione.
    Questa funzione realizza il collegamento tra Python e Prolog trasformando
    i risultati prodotti dagli algoritmi di Machine Learning in fatti logici.

    Funzionamento:
    1. Crea la cartella di destinazione se non esiste.
    2. Scrive nel file le soglie di business definite in config.py
       (budget, lingue, prezzo e review score).
    3. Converte ogni gioco nel predicato:
       gioco(Nome, Genere, Prezzo, Ore, Lingue, Multiplayer, Budget, ReviewScore).
    4. Converte il cluster previsto dal modello supervisionato nel predicato:
       predizione_commerciale(Gioco, Cluster).
    5. Converte la stima di rischio proveniente dalla componente probabilistica:
       rischio_bayesiano(Gioco, LivelloRischio).
    6. Salva il file .pl risultante e restituisce il percorso generato.

    Il file prodotto costituisce la base di fatti interrogata successivamente
    dal motore SWI-Prolog.
    """
    facts_path = Path(PROLOG_FACTS_PATH)
    facts_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% File generato da src/prolog_reasoning.py.",
        "",
        f"soglia_budget_standard({MAX_GRANT_BUDGET}).",
        f"soglia_budget_ridotta({REDUCED_GRANT_BUDGET}).",
        f"soglia_lingue_globale({MIN_GLOBAL_LANGUAGES}).",
        f"soglia_lingue_premium({MIN_PREMIUM_LANGUAGES}).",
        f"soglia_prezzo_alto({HIGH_PRICE_THRESHOLD}).",
        f"soglia_prezzo_basso({LOW_PRICE_THRESHOLD}).",
        f"soglia_review_bassa({LOW_REVIEW_SCORE_THRESHOLD}).",
        f"soglia_review_buona({GOOD_REVIEW_SCORE_THRESHOLD}).",
        "",
    ]

    # Prolog preferisce predicati contigui: prima tutti i gioco/8.
    for game in games:
        lines.append(
            "gioco("
            f"{game['name']}, "
            f"{game['genre']}, "
            f"{game['price']}, "
            f"{game['hours']}, "
            f"{game['languages']}, "
            f"{game['multiplayer']}, "
            f"{game['budget']}, "
            f"{game['review_score']}"
            ")."
        )

    lines.append("")
    # Poi tutte le predizioni commerciali derivate dal modello supervisionato.
    for game in games:
        lines.append(f"predizione_commerciale({game['name']}, {game['cluster']}).")

    lines.append("")
    # Infine il rischio derivato dalla componente bayesiana o da scenario business.
    for game in games:
        lines.append(f"rischio_bayesiano({game['name']}, {game['risk']}).")

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


def build_decision_rows(prolog) -> list[dict]:
    """Costruisce la tabella finale dei risultati decisionali.

    Rappresenta il punto di integrazione tra il motore logico Prolog
    e il layer analitico Python basato su Pandas.

    Funzionamento:
    1. Recupera tutti i giochi presenti nella Knowledge Base
       interrogando il predicato 'gioco/8'.
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
    game_names = query_all_values(prolog, "gioco(Nome, _, _, _, _, _, _, _)", "Nome")
    rows = []

    for game in game_names:
        # Verdetto principale: approvato, revisione o rifiutato.
        verdict = query_single_value(prolog, f"verdetto({game}, Verdetto)", "Verdetto")

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
    1. Rigenera il file dei fatti Prolog a partire dai dati correnti.
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
    - predizioni del clustering supervisionato;
    - valutazioni di rischio provenienti dalla rete bayesiana.
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
