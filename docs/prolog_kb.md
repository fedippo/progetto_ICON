# Ragionamento logico e Knowledge Base Prolog

## Obiettivo

La KB Prolog rappresenta il regolamento decisionale del publisher. Riceve in ingresso solo caratteristiche note o stimabili prima della pubblicazione:

- genere;
- prezzo;
- durata stimata;
- numero di lingue;
- presenza del multiplayer;
- budget richiesto.

Il risultato e un verdetto:

- `approvato`;
- `revisione`;
- `rifiutato`.

## File

- `kb/publisher_rules.pl`: regole stabili del publisher;
- `kb/generated_facts.pl`: fatti sui giochi analizzati, generati automaticamente;
- `src/prolog_reasoning.py`: integrazione Python-Prolog;
- `results/prolog_decisions.csv`: tabella dei verdetti.

`publisher_rules.pl` e l'unico file Prolog pensato per essere modificato manualmente. `generated_facts.pl` viene rigenerato da `src/prolog_reasoning.py` prima della consultazione della KB.

## Rappresentazione della conoscenza

I fatti principali sono:

```prolog
gioco(Nome, Genere, Prezzo, OreStimate, Lingue, Multiplayer, Budget).
```

Non vengono piu passati come fatti:

- review score;
- cluster predetto;
- rischio bayesiano;
- verdetto finale.

Questi elementi non sono caratteristiche pre-pubblicazione del gioco. La KB deve invece derivare internamente profilo commerciale, rischio e decisione.

## Generazione dei fatti

I fatti vengono scritti dal codice come predicati `gioco/7`. Ogni fatto descrive una proposta di gioco, non una decisione.

Esempio:

```prolog
gioco(hades_like, rpg, 24.99, 30, 8, 0, 120000).
```

Questa scelta evita che la KB venga usata come semplice tabella di output: Prolog riceve dati di partenza e applica regole.

## Regole principali

La KB contiene regole per:

- supporto globale: almeno 3 lingue;
- supporto premium: almeno 5 lingue;
- compatibilita del budget;
- prezzo alto o basso;
- coerenza tra genere e durata;
- profilo di successo, intermedio o rischio;
- rischio accettabile, medio o critico;
- violazioni bloccanti;
- approvazione;
- richiesta di revisione;
- rifiuto.

## Esempi di ragionamento

Un gioco puo essere approvato se:

- ha supporto premium;
- ha budget compatibile;
- la durata stimata e coerente con il genere;
- il prezzo non e alto;
- non presenta un profilo di rischio.

Un gioco puo richiedere revisione se:

- e localizzato;
- e coerente con il genere;
- ha budget compatibile;
- presenta rischio medio, ad esempio prezzo alto ma senza altre violazioni.

Un gioco viene rifiutato se presenta violazioni bloccanti, ad esempio:

- localizzazione insufficiente;
- incoerenza tra genere e durata;
- budget eccessivo;
- prezzo alto combinato con multiplayer.

## Perche non e semplice pattern matching

La KB non si limita a recuperare fatti espliciti. Le decisioni derivano da catene di regole:

- `verdetto/2` dipende da `approva_finanziamento/1`, `richiede_revisione/1` e `rifiuta_finanziamento/1`;
- `approva_finanziamento/1` usa `profilo_successo/1` e `rischio_accettabile/1`;
- `profilo_successo/1`, `profilo_intermedio/1` e `profilo_rischio/1` sono derivati da prezzo, lingue, budget, multiplayer e coerenza genere-durata;
- `violazione_bloccante/2` astrae motivazioni diverse in un unico meccanismo di rifiuto.

In questo modo il ragionamento logico produce nuova conoscenza a partire da fatti pre-lancio.

## Complessita

Il ragionamento Prolog avviene tramite risoluzione SLD su clausole di Horn. Nel nostro caso il numero di fatti e regole e limitato, quindi le query hanno costo contenuto.

In generale, il costo cresce con:

- numero di giochi rappresentati come fatti;
- numero di regole alternative per ciascun verdetto;
- presenza di negazione come fallimento;
- numero di condizioni concatenate nelle regole.

La KB e progettata per mantenere le regole modulari: aggiungere nuovi vincoli del publisher richiede nuove clausole, senza modificare la pipeline di apprendimento.
