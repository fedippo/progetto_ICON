# Ragionamento logico e Knowledge Base Prolog

## Obiettivo

La KB Prolog rappresenta il regolamento decisionale del publisher. Riceve in ingresso:

- dati del gioco;
- cluster predetto dal sistema di apprendimento;
- livello di rischio stimato dalla componente bayesiana.

Il risultato e un verdetto:

- `approvato`;
- `revisione`;
- `rifiutato`.

## File

- `kb/publisher_rules.pl`: regole stabili del publisher;
- `kb/generated_facts.pl`: fatti sui giochi analizzati, generati automaticamente;
- `src/prolog_reasoning.py`: integrazione Python-Prolog;
- `results/prolog_decisions.csv`: tabella dei verdetti.

`publisher_rules.pl` e l'unico file Prolog pensato per essere modificato manualmente. `generated_facts.pl` viene rigenerato da `src/prolog_reasoning.py` prima della consultazione della KB, in modo da mantenere i fatti ordinati e coerenti.

## Rappresentazione della conoscenza

I fatti principali sono:

```prolog
gioco(Nome, Genere, Prezzo, Ore, Lingue, Multiplayer, Budget, ReviewScoreStimato).
predizione_commerciale(Nome, Cluster).
rischio_bayesiano(Nome, Rischio).
```

Questa rappresentazione collega tre livelli:

- dati commerciali osservabili;
- output del machine learning;
- rischio probabilistico.

## Generazione dei fatti

I fatti vengono scritti dal codice in tre blocchi:

1. tutti i fatti `gioco/8`;
2. tutti i fatti `predizione_commerciale/2`;
3. tutti i fatti `rischio_bayesiano/2`.

Questa scelta evita warning Prolog sui predicati non contigui e separa la conoscenza stabile del publisher dai dati generati dalla pipeline.

## Regole principali

La KB contiene regole per:

- supporto globale: almeno 3 lingue;
- supporto premium: almeno 5 lingue;
- compatibilita del budget;
- prezzo alto o basso;
- review score basso o buono;
- coerenza tra genere e durata;
- rischio critico;
- violazioni bloccanti;
- approvazione;
- richiesta di revisione;
- rifiuto.

## Esempi di ragionamento

Un gioco puo essere approvato se:

- il cluster predetto e di successo;
- ha supporto globale;
- il budget e compatibile;
- la durata e coerente con il genere;
- il rischio bayesiano e accettabile.

Un gioco intermedio puo comunque essere approvato solo se:

- ha supporto premium;
- ha budget ridotto;
- ha review score stimato buono;
- non presenta incoerenze.

Un gioco viene rifiutato se presenta violazioni bloccanti, ad esempio:

- localizzazione insufficiente;
- rischio critico;
- incoerenza tra genere e durata.

## Perche non e semplice pattern matching

La KB non si limita a recuperare fatti espliciti. Le decisioni derivano da catene di regole:

- `verdetto/2` dipende da `approva_finanziamento/1`, `richiede_revisione/1` e `rifiuta_finanziamento/1`;
- `approva_finanziamento/1` combina cluster ML, budget, lingue, coerenza genere-durata e rischio bayesiano;
- `violazione_bloccante/2` astrae motivazioni diverse in un unico meccanismo di rifiuto;
- `rischio_accettabile/1` e `rischio_critico/1` combinano rischio probabilistico, prezzo e review score.

In questo modo il ragionamento logico integra output numerici e probabilistici in regole aziendali deterministiche.

## Complessita

Il ragionamento Prolog avviene tramite risoluzione SLD su clausole di Horn. Nel nostro caso il numero di fatti e regole e limitato, quindi le query hanno costo contenuto.

In generale, il costo cresce con:

- numero di giochi rappresentati come fatti;
- numero di regole alternative per ciascun verdetto;
- presenza di negazione come fallimento, usata per verificare assenza di violazioni o condizioni non soddisfatte.

La KB e progettata per mantenere le regole modulari: aggiungere nuovi vincoli del publisher richiede nuove clausole, senza modificare la pipeline di apprendimento.
