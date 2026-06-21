"""Integrazione Python-Prolog per le decisioni di finanziamento del publisher."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import PROLOG_DECISIONS_PATH, PROLOG_FACTS_PATH, PROLOG_RULES_PATH


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
    # Caso intermedio: buon supporto lingue, ma review non abbastanza alta per approvazione.
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
    """Genera fatti Prolog ordinati a partire da record strutturati."""
    facts_path = Path(PROLOG_FACTS_PATH)
    facts_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% File generato da src/prolog_reasoning.py. Non modificare manualmente.",
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
    """Importa PySwip solo quando serve, perche richiede anche SWI-Prolog installato."""
    try:
        from pyswip import Prolog
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Missing pyswip dependency. Install requirements.txt and SWI-Prolog before running this script."
        ) from exc
    return Prolog


def consult_knowledge_base():
    """Carica le regole Prolog e i fatti generati."""
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
    """Restituisce il primo valore associato a una variabile Prolog."""
    results = list(prolog.query(query))
    if not results:
        return None
    return str(results[0][variable])


def query_all_values(prolog, query: str, variable: str) -> list[str]:
    """Restituisce tutti i valori associati a una variabile Prolog."""
    return [str(row[variable]) for row in prolog.query(query)]


def unique_preserving_order(values: list[str]) -> list[str]:
    """Rimuove duplicati senza cambiare l'ordine dei risultati Prolog."""
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def build_decision_rows(prolog) -> list[dict]:
    """Interroga verdetti e violazioni bloccanti per ogni gioco noto."""
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
    """Esegue il ragionamento Prolog e salva la tabella decisionale."""
    # I fatti sono rigenerati a ogni run per evitare modifiche manuali incoerenti.
    generate_prolog_facts()
    prolog = consult_knowledge_base()
    decisions_df = pd.DataFrame(build_decision_rows(prolog))
    Path(PROLOG_DECISIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    decisions_df.to_csv(PROLOG_DECISIONS_PATH, index=False)
    return decisions_df


if __name__ == "__main__":
    decisions = run_prolog_reasoning()
    print(decisions.to_string(index=False))
