:- dynamic gioco/6.
:- dynamic predizione_commerciale/2.
:- dynamic rischio_bayesiano/2.
:- dynamic probabilita_recensioni_basse/2.
:- dynamic soglia_prezzo_alto/1.
:- dynamic soglia_prezzo_basso/1.
:- dynamic soglia_lingue_globale/1.
:- dynamic soglia_lingue_premium/1.

% gioco(Nome, Genere, Prezzo, OreStimate, Lingue, Multiplayer).
% predizione_commerciale(Nome, Cluster) e generato dal miglior modello supervisionato.
% rischio_bayesiano(Nome, Rischio) e stimato dalla rete bayesiana.
% Tutti i fatti descrivono informazioni note o stimabili prima della pubblicazione.

cluster_successo(cluster_0).
cluster_intermedio(cluster_2).
cluster_rischio(cluster_1).

genere_lungo(rpg).
genere_lungo(strategy).
genere_lungo(simulation).

genere_breve(puzzle).
genere_breve(casual).

supporto_globale(Gioco) :-
    gioco(Gioco, _, _, _, Lingue, _),
    soglia_lingue_globale(Soglia),
    Lingue >= Soglia.

supporto_premium(Gioco) :-
    gioco(Gioco, _, _, _, Lingue, _),
    soglia_lingue_premium(Soglia),
    Lingue >= Soglia.

prezzo_alto(Gioco) :-
    gioco(Gioco, _, Prezzo, _, _, _),
    soglia_prezzo_alto(Soglia),
    Prezzo >= Soglia.

prezzo_basso(Gioco) :-
    gioco(Gioco, _, Prezzo, _, _, _),
    soglia_prezzo_basso(Soglia),
    Prezzo =< Soglia.

coerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, Ore, _, _),
    genere_lungo(Genere),
    Ore >= 15.

coerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, Ore, _, _),
    genere_breve(Genere),
    Ore =< 25.

coerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, _, _, _),
    \+ genere_lungo(Genere),
    \+ genere_breve(Genere).

incoerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, Ore, _, _),
    genere_lungo(Genere),
    Ore < 15.

incoerenza_genere(Gioco) :-
    gioco(Gioco, Genere, _, Ore, _, _),
    genere_breve(Genere),
    Ore > 25.

multiplayer_presente(Gioco) :-
    gioco(Gioco, _, _, _, _, 1).

profilo_successo(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_successo(Cluster),
    supporto_premium(Gioco),
    coerenza_genere(Gioco),
    \+ prezzo_alto(Gioco).

profilo_intermedio(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_intermedio(Cluster),
    supporto_globale(Gioco),
    coerenza_genere(Gioco),
    \+ profilo_successo(Gioco),
    \+ profilo_rischio(Gioco).

profilo_rischio(Gioco) :-
    predizione_commerciale(Gioco, Cluster),
    cluster_rischio(Cluster).

profilo_rischio(Gioco) :-
    rischio_bayesiano(Gioco, alto).

profilo_rischio(Gioco) :-
    \+ supporto_globale(Gioco).

profilo_rischio(Gioco) :-
    incoerenza_genere(Gioco).

profilo_rischio(Gioco) :-
    prezzo_alto(Gioco),
    multiplayer_presente(Gioco).

rischio_medio(Gioco) :-
    rischio_bayesiano(Gioco, medio),
    \+ profilo_rischio(Gioco).

rischio_medio(Gioco) :-
    prezzo_alto(Gioco),
    supporto_premium(Gioco),
    \+ multiplayer_presente(Gioco),
    coerenza_genere(Gioco).

rischio_medio(Gioco) :-
    profilo_intermedio(Gioco).

rischio_accettabile(Gioco) :-
    \+ profilo_rischio(Gioco),
    \+ rischio_medio(Gioco).

rischio_critico(Gioco) :-
    profilo_rischio(Gioco).

violazione_bloccante(Gioco, localizzazione_insufficiente) :-
    \+ supporto_globale(Gioco).

violazione_bloccante(Gioco, incoerenza_genere) :-
    incoerenza_genere(Gioco).

violazione_bloccante(Gioco, rischio_critico) :-
    rischio_bayesiano(Gioco, alto).

violazione_bloccante(Gioco, rischio_critico) :-
    prezzo_alto(Gioco),
    multiplayer_presente(Gioco).

approva_finanziamento(Gioco) :-
    profilo_successo(Gioco),
    rischio_accettabile(Gioco).

richiede_revisione(Gioco) :-
    rischio_medio(Gioco),
    \+ violazione_bloccante(Gioco, _).

richiede_revisione(Gioco) :-
    profilo_intermedio(Gioco),
    \+ violazione_bloccante(Gioco, _).

richiede_revisione(Gioco) :-
    prezzo_basso(Gioco),
    supporto_globale(Gioco),
    coerenza_genere(Gioco),
    \+ approva_finanziamento(Gioco).

rifiuta_finanziamento(Gioco) :-
    violazione_bloccante(Gioco, _).

rifiuta_finanziamento(Gioco) :-
    rischio_critico(Gioco),
    \+ richiede_revisione(Gioco).

verdetto(Gioco, approvato) :-
    approva_finanziamento(Gioco).

verdetto(Gioco, revisione) :-
    \+ approva_finanziamento(Gioco),
    richiede_revisione(Gioco).

verdetto(Gioco, rifiutato) :-
    \+ approva_finanziamento(Gioco),
    \+ richiede_revisione(Gioco),
    rifiuta_finanziamento(Gioco).
