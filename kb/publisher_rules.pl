:- dynamic gioco/8.
:- dynamic predizione_commerciale/2.
:- dynamic rischio_bayesiano/2.
:- dynamic soglia_budget_standard/1.
:- dynamic soglia_budget_ridotta/1.
:- dynamic soglia_prezzo_alto/1.
:- dynamic soglia_prezzo_basso/1.
:- dynamic soglia_review_bassa/1.
:- dynamic soglia_review_buona/1.
:- dynamic soglia_lingue_globale/1.
:- dynamic soglia_lingue_premium/1.

% gioco(Nome, Genere, Prezzo, Ore, Lingue, Multiplayer, Budget, ReviewScoreStimato).
% predizione_commerciale(Nome, Cluster).
% rischio_bayesiano(Nome, Rischio).

cluster_successo(cluster_0).
cluster_intermedio(cluster_2).
cluster_rischio(cluster_1).

genere_lungo(rpg).
genere_lungo(strategy).
genere_lungo(simulation).

genere_breve(puzzle).
genere_breve(casual).

supporto_globale(Gioco) :-
    gioco(Gioco, _, _, _, Lingue, _, _, _),
    soglia_lingue_globale(Soglia),
    Lingue >= Soglia.

supporto_premium(Gioco) :-
    gioco(Gioco, _, _, _, Lingue, _, _, _),
    soglia_lingue_premium(Soglia),
    Lingue >= Soglia.

budget_standard_compatibile(Gioco) :-
    gioco(Gioco, _, _, _, _, _, Budget, _),
    soglia_budget_standard(Soglia),
    Budget =< Soglia.

budget_ridotto_compatibile(Gioco) :-
    gioco(Gioco, _, _, _, _, _, Budget, _),
    soglia_budget_ridotta(Soglia),
    Budget =< Soglia.

prezzo_alto(Gioco) :-
    gioco(Gioco, _, Prezzo, _, _, _, _, _),
    soglia_prezzo_alto(Soglia),
    Prezzo >= Soglia.

prezzo_basso(Gioco) :-
    gioco(Gioco, _, Prezzo, _, _, _, _, _),
    soglia_prezzo_basso(Soglia),
    Prezzo =< Soglia.

review_bassa(Gioco) :-
    gioco(Gioco, _, _, _, _, _, _, ReviewScore),
    soglia_review_bassa(Soglia),
    ReviewScore < Soglia.

review_buona(Gioco) :-
    gioco(Gioco, _, _, _, _, _, _, ReviewScore),
    soglia_review_buona(Soglia),
    ReviewScore >= Soglia.

coerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, Ore, _, _, _, _),
    genere_lungo(Genere),
    Ore >= 15.

coerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, Ore, _, _, _, _),
    genere_breve(Genere),
    Ore =< 25.

coerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, _, _, _, _, _),
    \+ genere_lungo(Genere),
    \+ genere_breve(Genere).

incoerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, Ore, _, _, _, _),
    genere_lungo(Genere),
    Ore < 15.

incoerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, Ore, _, _, _, _),
    genere_breve(Genere),
    Ore > 25.

rischio_accettabile(Gioco) :-
    rischio_bayesiano(Gioco, basso).

rischio_accettabile(Gioco) :-
    rischio_bayesiano(Gioco, medio),
    \+ prezzo_alto(Gioco).

rischio_critico(Gioco) :-
    rischio_bayesiano(Gioco, alto).

rischio_critico(Gioco) :-
    prezzo_alto(Gioco),
    review_bassa(Gioco).

violazione_bloccante(Gioco, incoerenza_genere) :-
    incoerenza_genere(Gioco).

violazione_bloccante(Gioco, rischio_critico) :-
    rischio_critico(Gioco).

violazione_bloccante(Gioco, localizzazione_insufficiente) :-
    \+ supporto_globale(Gioco).

approva_finanziamento(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_successo(Cluster),
    supporto_globale(Gioco),
    budget_standard_compatibile(Gioco),
    coerenza_genere(Gioco),
    rischio_accettabile(Gioco).

approva_finanziamento(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_intermedio(Cluster),
    supporto_premium(Gioco),
    budget_ridotto_compatibile(Gioco),
    coerenza_genere(Gioco),
    review_buona(Gioco),
    rischio_accettabile(Gioco).

richiede_revisione(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_successo(Cluster),
    supporto_globale(Gioco),
    coerenza_genere(Gioco),
    rischio_bayesiano(Gioco, medio),
    prezzo_alto(Gioco).

richiede_revisione(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_intermedio(Cluster),
    supporto_globale(Gioco),
    coerenza_genere(Gioco),
    \+ budget_ridotto_compatibile(Gioco).

richiede_revisione(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_intermedio(Cluster),
    supporto_globale(Gioco),
    budget_ridotto_compatibile(Gioco),
    coerenza_genere(Gioco),
    \+ review_buona(Gioco),
    \+ rischio_critico(Gioco).

rifiuta_finanziamento(Gioco) :-
    violazione_bloccante(Gioco, _).

rifiuta_finanziamento(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_rischio(Cluster),
    \+ review_buona(Gioco).

verdetto(Gioco, approvato) :-
    approva_finanziamento(Gioco).

verdetto(Gioco, revisione) :-
    \+ approva_finanziamento(Gioco),
    richiede_revisione(Gioco).

verdetto(Gioco, rifiutato) :-
    \+ approva_finanziamento(Gioco),
    \+ richiede_revisione(Gioco),
    rifiuta_finanziamento(Gioco).
