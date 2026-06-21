# IndieLaunch Pad: sistema intelligente per la valutazione commerciale di videogiochi indie su Steam

## Indice

- Capitolo 0 - Introduzione
- Capitolo 1 - Creazione del dataset e preprocessing
- Capitolo 2 - Apprendimento non supervisionato
- Capitolo 3 - Apprendimento supervisionato
- Capitolo 4 - Ragionamento probabilistico e rete bayesiana
- Capitolo 5 - Ragionamento logico e Knowledge Base Prolog
- Conclusioni
- Sviluppi futuri
- Riferimenti

## Capitolo 0 - Introduzione

L'obiettivo del progetto e realizzare un sistema di supporto alle decisioni per un publisher di videogiochi indipendenti. Il sistema simula il processo con cui un'etichetta di pubblicazione valuta una proposta commerciale: il publisher deve decidere se finanziare il gioco, richiedere una revisione del progetto oppure rifiutare la proposta.

Il progetto non e pensato come semplice esercizio di classificazione su un dataset standard. La parte di apprendimento automatico viene inserita in una pipeline piu ampia, in cui il risultato dei modelli e usato da componenti di ragionamento probabilistico e logico.

La pipeline realizzata e:

```text
Dataset Steam
  -> preprocessing e feature engineering
  -> clustering KMeans
  -> classificazione supervisionata del profilo commerciale
  -> rete bayesiana per scenari incerti
  -> Knowledge Base Prolog
  -> verdetto finale del publisher
```

Il dominio scelto e quello dei giochi pubblicati su Steam. Sono stati esclusi testi liberi, descrizioni, recensioni testuali e immagini. Il sistema lavora su metadati tabulari, come prezzo, genere, recensioni aggregate, numero di lingue, playtime e categorie Steam. Questa scelta evita di spostare il progetto verso NLP, image recognition o recommender systems, mantenendolo centrato sui temi del corso: apprendimento, ragionamento probabilistico e Knowledge-Based Systems.

### Requisiti funzionali

Il sistema deve:

- costruire un dataset pulito a partire dal CSV Steam;
- individuare profili commerciali latenti tramite clustering;
- usare i cluster come target per addestrare modelli supervisionati;
- valutare i modelli con risultati mediati su piu run;
- modellare scenari incerti con una rete bayesiana;
- integrare output di ML e rischio probabilistico in una KB Prolog;
- produrre un verdetto finale interpretabile.

### Strumenti utilizzati

Il progetto e realizzato in Python. Le librerie principali sono:

- `pandas` e `numpy` per caricamento e manipolazione dei dati;
- `scikit-learn` per preprocessing, clustering e modelli supervisionati;
- `imbalanced-learn` per SMOTE;
- `matplotlib` e `seaborn` per grafici e visualizzazioni;
- `pgmpy` per la rete bayesiana;
- `pyswip` e SWI-Prolog per l'integrazione con Prolog.

### Struttura del progetto e avvio

La struttura operativa e:

| Cartella | Contenuto |
|---|---|
| `data/raw/` | dataset originale |
| `data/processed/` | dataset puliti, normalizzati, discretizzati e clusterizzati |
| `src/` | script Python della pipeline |
| `kb/` | regole Prolog e fatti generati |
| `results/` | metriche, grafici, query e verdetti |
| `docs/` | documentazione e relazione |

Gli script principali sono eseguiti nel seguente ordine:

```text
src/preprocessing.py
src/clustering.py
src/supervised_learning.py
src/bayesian_network.py
src/prolog_reasoning.py
```

La cartella `results/` contiene gli output sperimentali usati nella relazione. Questa separazione permette di distinguere codice, dati processati, conoscenza logica e risultati.

## Capitolo 1 - Creazione del dataset e preprocessing

### Dataset scelto

Il dataset usato e "Steam Games Dataset" di fronkongames, disponibile su Kaggle. La scelta e motivata dai seguenti aspetti:

- contiene un numero ampio di giochi Steam;
- fornisce dati tabulari gia adatti all'analisi quantitativa;
- include variabili commerciali utili per il problema del publisher;
- permette di evitare testo libero e immagini;
- puo essere campionato per restare compatibile con il vincolo di circa 25 ore di lavoro.

Il file originale contiene 122611 righe. Per il progetto non e stato usato l'intero dataset in tutte le fasi: dopo pulizia e filtri viene estratto un campione riproducibile di 5000 giochi. La scelta del campionamento e progettuale: consente di mantenere tempi di esecuzione contenuti e rende gestibili clustering, cross-validation e apprendimento della rete bayesiana.

### Correzione dell'intestazione

Durante la prima lettura del CSV e emersa un'anomalia: nell'intestazione compare il campo `DiscountDLC count`, mentre nelle righe dati `Discount` e `DLC count` sono due colonne separate. Se il file viene letto senza correzione, tutte le colonne successive risultano sfalsate.

La correzione viene gestita direttamente nel codice di preprocessing:

- si legge la riga di intestazione;
- quando viene trovato `DiscountDLC count`, viene sostituito con due colonne;
- le colonne successive tornano allineate;
- il file CSV originale non viene modificato manualmente.

Questa scelta rende la pipeline riproducibile: chi esegue il progetto puo partire dal dataset originale e ottenere lo stesso dataset processato.

### Feature selection e feature engineering

Dal dataset originale vengono selezionate e derivate le feature necessarie alle fasi successive. Le principali sono:

| Feature | Origine | Uso nel progetto |
|---|---|---|
| `Primary_Genre` | primo valore di `Genres` | clustering, classificazione, rete bayesiana, Prolog |
| `Price` | prezzo Steam | clustering, classificazione, rete bayesiana, Prolog |
| `Review_Count` | `Positive + Negative` | volume di interesse commerciale |
| `Review_Score_Pct` | `Positive / (Positive + Negative)` | ricezione utente |
| `Playtime_Hours` | playtime medio in minuti convertito in ore | durata stimata |
| `Languages_Count` | conteggio di `Supported languages` | vincolo di localizzazione |
| `Multiplayer` | derivato da `Categories` | feature commerciale e rischio |

`Review_Count` e `Review_Score_Pct` sono particolarmente importanti perche rappresentano due aspetti diversi: il primo misura il volume di attenzione ricevuta dal gioco, il secondo misura la qualita percepita dagli utenti.

Sono state escluse feature accessorie come piattaforme supportate, eta richiesta, raccomandazioni e stima dei possessori. La scelta riduce la complessita del progetto e mantiene solo variabili direttamente collegate alla decisione commerciale che il publisher deve prendere.

### Filtri applicati

Sono stati rimossi:

- record senza nome;
- record senza genere;
- giochi con meno di 20 recensioni totali;
- giochi con prezzo fuori dall'intervallo 0-100;
- duplicati su `AppID`.

Il filtro sulle recensioni evita che il clustering sia dominato da giochi con pochissime informazioni di mercato. Il filtro sul prezzo rimuove outlier non rappresentativi per il caso d'uso di un publisher indie.

### Normalizzazione e discretizzazione

Vengono prodotti tre dataset:

- `steam_games_clean.csv`: dataset pulito con feature derivate;
- `steam_games_normalized.csv`: dataset normalizzato;
- `steam_games_discretized.csv`: dataset discretizzato.

La normalizzazione viene usata per KMeans e SVM, poiche entrambi sono sensibili alla scala delle feature. Senza normalizzazione, `Review_Count` potrebbe dominare variabili come prezzo e durata.

La discretizzazione viene usata per la rete bayesiana. Variabili continue con molti valori distinti produrrebbero CPD sparse e poco gestibili. La discretizzazione in stati basso, medio e alto rende invece l'inferenza piu stabile e interpretabile.

## Capitolo 2 - Apprendimento non supervisionato

### Obiettivo del clustering

Il clustering serve a individuare profili commerciali latenti nel mercato Steam. Non viene usato come fase isolata, ma come passaggio intermedio:

- i cluster diventano il target della fase supervisionata;
- il cluster predetto viene poi usato nella KB Prolog;
- la rete bayesiana include `Cluster_Label` tra le variabili.

Questa integrazione rende il clustering strettamente utile al problema decisionale.

### Feature usate

Il KMeans viene applicato alle feature:

- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
- `Playtime_Hours`.

La scelta e legata al significato commerciale delle variabili: prezzo, ricezione, volume di recensioni e durata descrivono meglio di altre feature il profilo di mercato di un gioco.

### Scelta del numero di cluster

Il numero di cluster e stato valutato tramite Elbow Method per valori di `K` da 1 a 10. I risultati sono salvati in `results/kmeans_elbow.csv` e il grafico in `results/kmeans_elbow.png`.

Il valore scelto e `K = 3`. La scelta e coerente sia con la curva del gomito sia con l'obiettivo interpretativo del progetto: distinguere giochi con ricezione molto positiva, giochi intermedi e giochi a rischio commerciale piu alto.

Parametri principali:

- `K = 3`;
- `n_init = 10`;
- `max_iter = 300`;
- seed fisso per riproducibilita.

### Risultati del clustering

La tabella dei centroidi interpretati e:

| Cluster | Giochi | Prezzo medio | Review score medio | Recensioni mediane | Playtime medio | Interpretazione |
|---|---:|---:|---:|---:|---:|---|
| 0 | 2496 | 5.479 | 90.864 | 125 | 7.313 | ricezione molto positiva |
| 1 | 755 | 3.857 | 46.388 | 61 | 3.118 | rischio commerciale piu alto |
| 2 | 1749 | 5.524 | 72.394 | 98 | 7.768 | ricezione intermedia |

Il cluster 0 presenta la migliore ricezione media, con review score superiore al 90%. Il cluster 1 ha review score molto piu basso e minore playtime medio, quindi viene interpretato come profilo a maggiore rischio. Il cluster 2 rappresenta una fascia intermedia: non e un fallimento netto, ma neppure un profilo fortemente positivo.

Gli identificativi numerici dei cluster non hanno significato intrinseco: sono assegnati dall'algoritmo. Per questo motivo l'interpretazione viene fatta osservando le statistiche dei centroidi e viene mantenuta coerente nelle fasi successive del progetto.

L'output di questa fase e salvato nei dataset clusterizzati, in particolare `steam_games_normalized_clustered.csv`.

## Capitolo 3 - Apprendimento supervisionato

### Obiettivo

La fase supervisionata deve predire il profilo commerciale di un nuovo gioco, usando come target `Cluster_Label`. In questo modo il sistema puo stimare in anticipo se una proposta si avvicina a un profilo positivo, intermedio o rischioso.

### Feature e target

Il dataset usato e `steam_games_normalized_clustered.csv`. Il target e `Cluster_Label`.

Le feature usate sono:

- `Primary_Genre`;
- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
- `Playtime_Hours`;
- `Languages_Count`;
- `Multiplayer`.

`Primary_Genre` viene codificato tramite one-hot encoding dentro la pipeline, cosi la trasformazione avviene in modo coerente durante cross-validation e training.

### Modelli e iperparametri

Sono stati confrontati Random Forest e SVM.

Per Random Forest sono stati ricercati:

- `n_estimators`: 100, 200;
- `max_depth`: `None`, 10, 20;
- `min_samples_leaf`: 1, 3.

Per SVM sono stati ricercati:

- `C`: 0.1, 1, 10;
- `kernel`: `rbf`, `linear`;
- `gamma`: `scale`.

La griglia e stata mantenuta compatta per rispettare i tempi del progetto, ma sufficiente a confrontare modelli con capacita diverse.

### Valutazione

La valutazione usa `RepeatedStratifiedKFold` con:

- 5 fold;
- 3 ripetizioni;
- seed fisso.

Le metriche riportate sono:

- accuracy;
- precision macro;
- recall macro;
- F1 macro.

La scelta di metriche macro e motivata dalla presenza di classi non perfettamente bilanciate. In particolare, la distribuzione e:

| Cluster | Esempi | Percentuale |
|---|---:|---:|
| 0 | 2496 | 49.92% |
| 1 | 755 | 15.10% |
| 2 | 1749 | 34.98% |

Per questo motivo ogni modello viene valutato anche con SMOTE. SMOTE e inserito dentro la pipeline di cross-validation: viene applicato solo al training fold, evitando data leakage.

### Risultati

| Modello | SMOTE | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|
| Random Forest | No | 0.9979 +/- 0.0016 | 0.9973 +/- 0.0020 | 0.9969 +/- 0.0028 | 0.9971 +/- 0.0023 |
| Random Forest | Si | 0.9983 +/- 0.0009 | 0.9974 +/- 0.0015 | 0.9980 +/- 0.0014 | 0.9977 +/- 0.0014 |
| SVM | No | 0.9895 +/- 0.0022 | 0.9897 +/- 0.0022 | 0.9859 +/- 0.0037 | 0.9877 +/- 0.0028 |
| SVM | Si | 0.9890 +/- 0.0034 | 0.9841 +/- 0.0049 | 0.9901 +/- 0.0031 | 0.9870 +/- 0.0040 |

Migliori parametri trovati:

| Modello | SMOTE | Parametri migliori |
|---|---|---|
| Random Forest | No | `max_depth=None`, `min_samples_leaf=1`, `n_estimators=200` |
| Random Forest | Si | `max_depth=None`, `min_samples_leaf=3`, `n_estimators=100` |
| SVM | No | `C=10`, `gamma=scale`, `kernel=linear` |
| SVM | Si | `C=10`, `gamma=scale`, `kernel=linear` |

La Random Forest con SMOTE ottiene il miglior F1 macro, pari a 0.9977 +/- 0.0014. Il miglioramento rispetto alla versione senza SMOTE riguarda sia il valore medio sia la deviazione standard. Questo indica che il bilanciamento aiuta il modello a essere piu stabile sui diversi fold.

Per SVM, invece, SMOTE non porta un miglioramento complessivo: la recall macro aumenta, ma precision macro e F1 macro diminuiscono. La versione SVM senza SMOTE e quindi preferibile alla versione con SMOTE, ma resta inferiore alla Random Forest.

Il modello candidato per l'integrazione nel sistema decisionale e Random Forest con SMOTE.

## Capitolo 4 - Ragionamento probabilistico e rete bayesiana

### Obiettivo

La rete bayesiana viene usata per ragionare in condizioni di incertezza. Il classificatore supervisionato assegna un profilo commerciale, mentre la rete bayesiana permette di interrogare scenari parziali, ad esempio:

- cosa succede se il prezzo e alto?
- il multiplayer aumenta il rischio di recensioni basse?
- quale cluster e piu probabile dato un certo genere e un certo prezzo?

### Dataset e variabili

La rete usa `steam_games_discretized_clustered.csv`. Le variabili sono:

- `Primary_Genre`;
- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
- `Playtime_Hours`;
- `Languages_Count`;
- `Multiplayer`;
- `Cluster_Label`.

Le variabili numeriche sono discretizzate in stati 0, 1, 2, interpretati come basso, medio e alto. Questa discretizzazione evita problemi di memoria e CPD molto sparse.

### Apprendimento della struttura e dei parametri

La struttura viene appresa tramite HillClimbSearch con score BIC. E stato imposto `max_indegree = 3`, cosi ogni nodo puo avere al massimo tre genitori. La scelta limita la complessita della rete e rende piu leggibili le dipendenze apprese.

I parametri vengono appresi con l'estimatore discreto compatibile con la versione installata di `pgmpy`.

### Struttura appresa

| Sorgente | Destinazione |
|---|---|
| `Price` | `Languages_Count` |
| `Review_Score_Pct` | `Multiplayer` |
| `Multiplayer` | `Primary_Genre` |
| `Review_Count` | `Languages_Count` |
| `Review_Count` | `Multiplayer` |
| `Review_Count` | `Cluster_Label` |
| `Cluster_Label` | `Review_Score_Pct` |
| `Cluster_Label` | `Price` |
| `Playtime_Hours` | `Review_Count` |
| `Playtime_Hours` | `Price` |

Gli archi piu utili per il problema sono `Review_Count -> Cluster_Label`, `Cluster_Label -> Review_Score_Pct` e `Playtime_Hours -> Review_Count`, perche collegano comportamento commerciale, ricezione e durata.

### Inferenza

Risultati principali:

| Query | Stato | Probabilita |
|---|---:|---:|
| `P(Cluster_Label | Primary_Genre=0, Price=2)` | 0 | 0.5365 |
| `P(Cluster_Label | Primary_Genre=0, Price=2)` | 1 | 0.1090 |
| `P(Cluster_Label | Primary_Genre=0, Price=2)` | 2 | 0.3544 |
| `P(Cluster_Label | Price=0, Playtime_Hours=1)` | 0 | 0.3586 |
| `P(Cluster_Label | Price=0, Playtime_Hours=1)` | 1 | 0.2137 |
| `P(Cluster_Label | Price=0, Playtime_Hours=1)` | 2 | 0.4277 |
| `P(Review_Score_Pct | Price=2, Multiplayer=1)` | 0 | 0.3661 |
| `P(Review_Score_Pct | Price=2, Multiplayer=1)` | 1 | 0.3413 |
| `P(Review_Score_Pct | Price=2, Multiplayer=1)` | 2 | 0.2925 |

La query sul prezzo alto e multiplayer e rilevante per il publisher: la probabilita di review score basso e circa 36.61%, maggiore della probabilita di review score alto. Questo risultato suggerisce che, in uno scenario con prezzo alto e componente multiplayer, il publisher dovrebbe valutare con cautela aspettative dell'utente e posizionamento di prezzo.

## Capitolo 5 - Ragionamento logico e Knowledge Base Prolog

### Obiettivo

La KB Prolog rappresenta il regolamento interno del publisher. Il suo compito non e ripetere la classificazione, ma applicare regole decisionali deterministiche combinando:

- dati della proposta di gioco;
- cluster predetto dal modello supervisionato;
- rischio stimato dalla rete bayesiana.

Il sistema produce tre possibili verdetti:

- `approvato`;
- `revisione`;
- `rifiutato`.

### Rappresentazione della conoscenza

I fatti principali sono:

```prolog
gioco(Nome, Genere, Prezzo, Ore, Lingue, Multiplayer, Budget, ReviewScoreStimato).
predizione_commerciale(Nome, Cluster).
rischio_bayesiano(Nome, Rischio).
```

Le regole sono mantenute in `kb/publisher_rules.pl`. I fatti sono generati automaticamente in `kb/generated_facts.pl` dallo script `src/prolog_reasoning.py`. Questa separazione evita di modificare manualmente file generati e distingue la conoscenza stabile del dominio dai dati prodotti dalla pipeline.

### Regole implementate

La KB contiene regole per:

- supporto globale: almeno 3 lingue;
- supporto premium: almeno 5 lingue;
- compatibilita del budget standard e ridotto;
- prezzo alto o basso;
- review score basso o buono;
- coerenza tra genere e durata;
- rischio accettabile o critico;
- violazioni bloccanti;
- approvazione, revisione e rifiuto.

Esempio di logica decisionale:

- un gioco in cluster di successo puo essere approvato se ha supporto globale, budget compatibile, coerenza genere-durata e rischio accettabile;
- un gioco intermedio puo essere approvato solo con supporto premium, budget ridotto, review score buono e nessuna violazione;
- un gioco con rischio critico, localizzazione insufficiente o incoerenza di genere viene rifiutato.

### Perche la KB non e pattern matching

La KB non si limita a cercare fatti esplicitamente presenti. Il verdetto viene derivato da catene di regole. Ad esempio:

- `verdetto/2` dipende da `approva_finanziamento/1`, `richiede_revisione/1` e `rifiuta_finanziamento/1`;
- `approva_finanziamento/1` combina cluster, lingue, budget, durata e rischio;
- `violazione_bloccante/2` astrae cause diverse di rifiuto;
- `rischio_critico/1` puo derivare sia da rischio bayesiano alto sia dalla combinazione di prezzo alto e review score basso.

Questa struttura integra output numerici e probabilistici in regole aziendali interpretabili.

### Risultati dimostrativi

| Gioco | Verdetto | Motivi bloccanti |
|---|---|---|
| `hades_like` | approvato |  |
| `short_puzzle_deluxe` | revisione |  |
| `underlocalized_action` | rifiutato | localizzazione insufficiente |
| `overpriced_multiplayer_rpg` | rifiutato | incoerenza genere; rischio critico |

I quattro casi sono stati scelti per coprire comportamenti diversi:

- approvazione diretta di un gioco coerente e promettente;
- richiesta di revisione per un gioco intermedio;
- rifiuto per vincolo commerciale non soddisfatto;
- rifiuto per combinazione di incoerenza e rischio.

### Complessita della KB

Il ragionamento Prolog usa risoluzione SLD su clausole di Horn. Nel progetto il numero di fatti e regole e contenuto, quindi il costo delle query e basso. Tuttavia, in generale, la complessita cresce con:

- numero di giochi rappresentati;
- numero di regole alternative per ciascun verdetto;
- uso di negazione come fallimento;
- numero di condizioni concatenate nelle regole.

La KB e modulare: nuovi vincoli del publisher possono essere aggiunti introducendo nuove clausole, senza modificare clustering, classificazione o rete bayesiana.

## Conclusioni

Il progetto realizza un sistema integrato di supporto alle decisioni. Il clustering produce profili commerciali; la classificazione supervisionata predice il profilo di nuove proposte; la rete bayesiana ragiona su scenari incerti; la KB Prolog applica vincoli aziendali espliciti per produrre un verdetto.

La valutazione supervisionata rispetta il requisito di non basarsi su un singolo run: i risultati sono mediati su cross-validation ripetuta e includono deviazione standard. La parte Prolog evita una KB usata come semplice database, perche il ragionamento finale dipende da regole multicriterio e da vincoli di coerenza.

Il modello candidato per la predizione del cluster e Random Forest con SMOTE, che raggiunge il miglior F1 macro e la maggiore stabilita. La rete bayesiana fornisce invece un supporto probabilistico utile per valutare rischio di prezzo e ricezione utente. La KB conclude il processo trasformando questi risultati in una decisione interpretabile.

## Sviluppi futuri

Possibili estensioni:

- generare i fatti Prolog direttamente da una proposta inserita dall'utente;
- collegare automaticamente le probabilita bayesiane a livelli di rischio basso, medio e alto;
- aggiungere una piccola interfaccia per simulare scenari di investimento;
- testare la stabilita dei cluster su campioni diversi;
- introdurre ulteriori regole commerciali, ad esempio vincoli di piattaforma o requisiti per festival indie;
- esportare la relazione finale in formato PDF.

## Riferimenti

- Dataset: Kaggle, "Steam Games Dataset", fronkongames.
- Python: pandas, numpy, scikit-learn, imbalanced-learn.
- Rete bayesiana: pgmpy.
- Ragionamento logico: SWI-Prolog e pyswip.
- Output sperimentali: cartella `results/`.
