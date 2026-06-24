# IndieLaunch Pad: sistema intelligente per la valutazione commerciale di videogiochi indie su Steam

## Gruppo di lavoro

- federico ippino, 797726, f.ippino@studenti.uniba.it

<URL repo associato, contenente il materiale completo>

AA 2025-26

# Introduzione

Il progetto realizza un sistema di supporto alle decisioni per un publisher di videogiochi indipendenti. L’obiettivo è simulare il processo con cui un’etichetta di pubblicazione valuta una proposta commerciale e decide se finanziare il gioco, chiedere una revisione oppure rifiutare la proposta.

Il lavoro non è pensato come una semplice esercitazione di classificazione. L’apprendimento automatico è inserito in una pipeline più ampia, nella quale il risultato dei modelli viene usato da componenti di ragionamento probabilistico e logico. In questo modo il progetto integra più temi: apprendimento non supervisionato, apprendimento supervisionato, ragionamento su incertezza e rappresentazione della conoscenza.

La pipeline realizzata è:

```text
Dataset Steam
  -> preprocessing e feature engineering
  -> clustering KMeans
  -> classificazione supervisionata del profilo commerciale
  -> rete bayesiana per scenari incerti
  -> Knowledge Base Prolog
  -> verdetto finale del publisher
```

Il dominio scelto è quello dei giochi pubblicati su Steam. Sono stati esclusi testi liberi, recensioni testuali e immagini, così da mantenere il progetto centrato sui metadati tabellari e non spostarlo verso NLP o computer vision. Le variabili usate riguardano prezzo, genere, recensioni aggregate, numero di lingue, durata media di gioco e categorie Steam.

## Elenco argomenti di interesse

Gli argomenti trattati nel progetto sono:

- apprendimento non supervisionato e clustering;
- apprendimento supervisionato su dati tabulari;
- valutazione di modelli con cross-validation e metriche macro;
- modellazione probabilistica con rete bayesiana;
- ragionamento logico con Knowledge Base Prolog;
- integrazione tra moduli di apprendimento e ragionamento.

# Argomento 1 - Creazione del dataset e preprocessing

## Sommario

Il primo modulo costruisce un dataset pulito e coerente a partire dal CSV originale di Steam. Lo scopo è preparare i dati per le fasi successive della pipeline, riducendo errori di parsing, rimuovendo record poco affidabili e creando feature utili sia al clustering sia ai modelli supervisionati e probabilistici.

## Strumenti utilizzati

Il progetto è realizzato in Python. Le librerie principali sono:

- `pandas` e `numpy` per caricamento e manipolazione dei dati;
- `scikit-learn` per normalizzazione, discretizzazione e clustering;
- `imbalanced-learn` per SMOTE;
- `matplotlib` e `seaborn` per grafici e visualizzazioni.

Il dataset usato è “Steam Games Dataset” di fronkongames, disponibile su Kaggle. Il file originale contiene 122611 righe. Per il progetto non è stato usato l’intero dataset in tutte le fasi: dopo pulizia e filtri viene estratto un campione riproducibile di 5000 giochi, così da mantenere tempi di esecuzione contenuti e rendere gestibili clustering, cross-validation e apprendimento della rete bayesiana.

## Decisioni di Progetto

### Correzione dell’intestazione

Durante la lettura del CSV è emersa un’anomalia: nell’intestazione compare il campo `DiscountDLC count`, mentre nelle righe dati `Discount` e `DLC count` sono due colonne separate. Se il file viene letto senza correzione, le colonne successive risultano sfalsate.

La correzione viene gestita direttamente nel codice di preprocessing. La riga di intestazione viene letta e, quando viene trovato `DiscountDLC count`, il campo viene sostituito con due colonne distinte. In questo modo il file originale non viene modificato manualmente e la pipeline resta riproducibile.

### Feature selection e feature engineering

Dal dataset originale vengono selezionate e derivate le feature necessarie alle fasi successive:

| Feature | Origine | Uso nel progetto |
|---|---|---|
| `Primary_Genre` | primo valore di `Genres` | clustering, classificazione, rete bayesiana, Prolog |
| `Price` | prezzo Steam | clustering, classificazione, rete bayesiana, Prolog |
| `Review_Count` | `Positive + Negative` | volume di interesse commerciale |
| `Review_Score_Pct` | `Positive / (Positive + Negative)` | ricezione utente |
| `Playtime_Hours` | playtime medio in minuti convertito in ore | durata stimata |
| `Languages_Count` | conteggio di `Supported languages` | vincolo di localizzazione |
| `Multiplayer` | derivato da `Categories` | feature commerciale e rischio |

`Review_Count` e `Review_Score_Pct` sono particolarmente importanti perché rappresentano due aspetti diversi: il primo misura il volume di attenzione ricevuta dal gioco, il secondo misura la qualità percepita dagli utenti.

Sono state escluse feature accessorie come piattaforme supportate, età richiesta, raccomandazioni e stima dei possessori. La scelta riduce la complessità del progetto e mantiene solo variabili direttamente collegate alla decisione commerciale.

### Filtri applicati

Sono stati rimossi:

- record senza nome;
- record senza genere;
- giochi con meno di 20 recensioni totali;
- giochi con prezzo fuori dall’intervallo 0-100;
- duplicati su `AppID`.

Il filtro sulle recensioni evita che il clustering sia dominato da giochi con pochissime informazioni di mercato. Il filtro sul prezzo rimuove outlier non rappresentativi per il caso d’uso di un publisher indie.

### Normalizzazione e discretizzazione

Vengono prodotti tre dataset:

- `steam_games_clean.csv`: dataset pulito con feature derivate;
- `steam_games_normalized.csv`: dataset normalizzato;
- `steam_games_discretized.csv`: dataset discretizzato.

La normalizzazione viene usata per KMeans e SVM, poiché entrambi sono sensibili alla scala delle feature. Senza normalizzazione, `Review_Count` potrebbe dominare variabili come prezzo e durata.

La discretizzazione viene usata per la rete bayesiana. Variabili continue con molti valori distinti produrrebbero CPD sparse e poco gestibili. La discretizzazione in stati basso, medio e alto rende invece l’inferenza più stabile e interpretabile. La normalizzazione è implementata con `MinMaxScaler`, mentre la discretizzazione usa `KBinsDiscretizer` con strategia `quantile`, scelta perché distribuisce meglio bin molto sbilanciati, come quelli delle recensioni.

## Valutazione

La qualità del preprocessing è verificata indirettamente nelle fasi successive: il clustering risulta stabile e interpretabile, il classificatore supervisionato raggiunge prestazioni elevate e la rete bayesiana produce query leggibili e coerenti con le variabili discretizzate. I tre dataset generati consentono di separare chiaramente i bisogni dei moduli della pipeline.

# Argomento 2 - Apprendimento non supervisionato

## Sommario

Il clustering serve a individuare profili commerciali latenti nel mercato Steam. Non viene usato come fase isolata, ma come passaggio intermedio: i cluster diventano il target della fase supervisionata, il cluster predetto viene poi usato nella KB Prolog e la rete bayesiana include `Cluster_Label` tra le variabili.

## Strumenti utilizzati

Il clustering è implementato con `KMeans` di `scikit-learn`. Lo script lavora sul dataset normalizzato e usa le feature:

- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
- `Playtime_Hours`.

La scelta privilegia variabili con significato commerciale diretto e con buona interpretabillità dei centroidi.

## Decisioni di Progetto

KMeans è stato scelto perché produce centroidi facilmente interpretabili e si adatta bene a dati numerici normalizzati. Sono state considerate anche alternative come clustering gerarchico e DBSCAN, ma sono state scartate perché meno immediate da integrare nella pipeline complessiva e meno comode da riutilizzare come target supervisionato.

Il numero di cluster è stato valutato con l’Elbow Method per valori di `K` da 1 a 10. La scelta finale è `K = 3`, coerente sia con la curva del gomito sia con l’obiettivo interpretativo del progetto: distinguere giochi con ricezione positiva, giochi intermedi e giochi a rischio commerciale più alto.

Parametri principali:

- `K = 3`;
- `n_init = 10`;
- `max_iter = 300`;
- seed fisso per riproducibilità.

Nel codice, la fase è implementata in `src/clustering.py`. Lo script carica i dataset pulito, normalizzato e discretizzato; esegue l’Elbow Method sul dataset normalizzato; applica KMeans con `SELECTED_K = 3`; infine copia la colonna `Cluster_Label` in tutti i dataset processati, evitando disallineamenti tra moduli diversi.

### Risultati del clustering

| Cluster | Giochi | Prezzo medio | Review score medio | Recensioni mediane | Playtime medio | Interpretazione |
|---|---:|---:|---:|---:|---:|---|
| 0 | 2437 | 6.620 | 89.993 | 639 | 18.380 | ricezione molto positiva |
| 1 | 757 | 4.215 | 48.602 | 143 | 6.374 | rischio commerciale più alto |
| 2 | 1806 | 5.243 | 73.197 | 303 | 10.170 | ricezione intermedia |

Il cluster 0 presenta la migliore ricezione media, con review score superiore al 90%. Il cluster 1 ha review score molto più basso e minore playtime medio, quindi viene interpretato come profilo a maggiore rischio. Il cluster 2 rappresenta una fascia intermedia.

## Valutazione

La validazione del clustering è di tipo interpretativo: i centroidi risultano coerenti con il significato commerciale delle feature e il numero di cluster è sufficientemente compatto da poter essere riutilizzato come target supervisionato e come variabile della rete bayesiana.

# Argomento 3 - Apprendimento supervisionato

## Sommario

La fase supervisionata deve predire il profilo commerciale di un gioco usando come target `Cluster_Label`. Lo scopo non è prevedere il successo commerciale da zero, ma imparare a riconoscere automaticamente i profili latenti scoperti dal clustering.

## Strumenti utilizzati

Il dataset usato è `steam_games_normalized_clustered.csv`. Le feature impiegate sono:

- `Primary_Genre`;
- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
- `Playtime_Hours`;
- `Languages_Count`;
- `Multiplayer`.

`Primary_Genre` viene codificato tramite one-hot encoding dentro la pipeline. I modelli confrontati sono `Random Forest` e `SVM`. L’ottimizzazione degli iperparametri viene eseguita con `GridSearchCV`, mentre la valutazione usa `RepeatedStratifiedKFold`.

## Decisioni di Progetto

Sono stati confrontati Random Forest e SVM per valutare due approcci differenti alla classificazione di dati tabulari. Random Forest è un modello ensemble robusto su feature eterogenee; SVM consente invece di verificare se una separazione geometrica nello spazio normalizzato è sufficiente a distinguere i cluster.

Non sono stati scelti modelli neurali perché il dataset è tabulare, di dimensione gestibile e con poche feature. Non è stata scelta una regressione logistica come modello principale perché i cluster non sono necessariamente separabili linearmente.

Per Random Forest sono stati ricercati:

- `n_estimators`: 100, 200;
- `max_depth`: `None`, 10, 20;
- `min_samples_leaf`: 1, 3.

Per SVM sono stati ricercati:

- `C`: 0.1, 1, 10;
- `kernel`: `rbf`, `linear`;
- `gamma`: `scale`.

La griglia è stata mantenuta compatta per rispettare i tempi del progetto, ma sufficiente a confrontare modelli con capacità diverse.

Per il bilanciamento delle classi viene usato SMOTE dentro la pipeline di cross-validation. In questo modo l’oversampling avviene solo sul training fold, evitando data leakage.

La valutazione utilizza:

- 5 fold;
- 3 ripetizioni;
- seed fisso.

Le metriche riportate sono accuracy, precision macro, recall macro e F1 macro. La scelta di metriche macro è motivata dalla presenza di classi non perfettamente bilanciate.

### Risultati

| Modello | SMOTE | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|
| Random Forest | No | 0.9929 +/- 0.0018 | 0.9930 +/- 0.0018 | 0.9945 +/- 0.0014 | 0.9937 +/- 0.0015 |
| Random Forest | Si | 0.9933 +/- 0.0020 | 0.9933 +/- 0.0023 | 0.9950 +/- 0.0015 | 0.9941 +/- 0.0019 |
| SVM | No | 0.9878 +/- 0.0037 | 0.9879 +/- 0.0046 | 0.9869 +/- 0.0042 | 0.9874 +/- 0.0042 |
| SVM | Si | 0.9804 +/- 0.0038 | 0.9735 +/- 0.0054 | 0.9837 +/- 0.0037 | 0.9784 +/- 0.0045 |

Migliori parametri trovati:

| Modello | SMOTE | Parametri migliori |
|---|---|---|
| Random Forest | No | `max_depth=20`, `min_samples_leaf=1`, `n_estimators=200` |
| Random Forest | Si | `max_depth=10`, `min_samples_leaf=1`, `n_estimators=200` |
| SVM | No | `C=10`, `gamma=scale`, `kernel=linear` |
| SVM | Si | `C=10`, `gamma=scale`, `kernel=linear` |

La Random Forest con SMOTE ottiene il miglior F1 macro, pari a 0.9941 +/- 0.0019. Per SVM, SMOTE non porta un miglioramento complessivo: la recall macro aumenta, ma precision macro e F1 macro diminuiscono. Il modello candidato per l’integrazione nel sistema decisionale è quindi Random Forest con SMOTE.

## Valutazione

Il risultato va interpretato con cautela: le metriche molto alte dipendono anche dal fatto che il target supervisionato è stato generato dal clustering sulle stesse feature commerciali. La fase supervisionata resta comunque utile, perché il suo scopo non è scoprire una ground truth esterna, ma imparare a replicare il profilo commerciale latente assegnato dalla fase non supervisionata.

# Argomento 4 - Ragionamento probabilistico e rete bayesiana

## Sommario

La rete bayesiana viene usata per ragionare in condizioni di incertezza. Il classificatore supervisionato assegna un profilo commerciale, mentre la rete bayesiana permette di interrogare scenari parziali e di studiare dipendenze probabilistiche tra variabili.

## Strumenti utilizzati

La rete usa `steam_games_discretized_clustered.csv`. Le variabili sono:

- `Primary_Genre`;
- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
- `Playtime_Hours`;
- `Languages_Count`;
- `Multiplayer`;
- `Cluster_Label`.

Le variabili numeriche sono discretizzate in stati 0, 1, 2, interpretati come basso, medio e alto. Per le variabili categoriche, come `Primary_Genre`, il codice esporta anche una legenda in `results/category_mappings.csv`.

L’implementazione usa `pgmpy`.

## Decisioni di Progetto

La struttura viene appresa tramite `HillClimbSearch` con score BIC. È stato imposto `max_indegree = 3`, così ogni nodo può avere al massimo tre genitori. La scelta limita la complessità della rete e rende più leggibili le dipendenze apprese.

È stata adottata una rete bayesiana discreta perché la discretizzazione delle variabili semplifica l’apprendimento delle CPD e rende più interpretabili i risultati. La struttura è stata appresa automaticamente per evitare di introdurre assunzioni manuali sulle dipendenze tra variabili.

Nel codice, la fase è implementata in `src/bayesian_network.py`. Lo script carica il dataset discretizzato e clusterizzato, apprende la struttura, stima le CPD e salva sia gli archi appresi sia le query in formato CSV.

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

Gli archi più utili per il problema sono `Review_Count -> Cluster_Label`, `Cluster_Label -> Review_Score_Pct` e `Playtime_Hours -> Review_Count`, perché collegano comportamento commerciale, ricezione e durata.

## Valutazione

I risultati principali dell’inferenza sono:

| Query | Stato | Probabilità |
|---|---:|---:|
| `P(Cluster_Label | Primary_Genre=1, Price=2)` | 0 | 0.5765 |
| `P(Cluster_Label | Primary_Genre=1, Price=2)` | 1 | 0.1014 |
| `P(Cluster_Label | Primary_Genre=1, Price=2)` | 2 | 0.3222 |
| `P(Cluster_Label | Price=0, Playtime_Hours=2)` | 0 | 0.3847 |
| `P(Cluster_Label | Price=0, Playtime_Hours=2)` | 1 | 0.2146 |
| `P(Cluster_Label | Price=0, Playtime_Hours=2)` | 2 | 0.4008 |
| `P(Review_Score_Pct | Price=2, Multiplayer=1)` | 0 | 0.3139 |
| `P(Review_Score_Pct | Price=2, Multiplayer=1)` | 1 | 0.3636 |
| `P(Review_Score_Pct | Price=2, Multiplayer=1)` | 2 | 0.3226 |

La legenda esportata in `results/category_mappings.csv` permette di leggere `Primary_Genre=1` come `Action`. Quindi la prima query valuta giochi Action con prezzo alto: il cluster più probabile è il cluster 0, cioè il profilo con ricezione molto positiva. La seconda query mostra che un gioco economico ma con playtime alto si distribuisce soprattutto tra cluster 2 e cluster 0. La terza query indica che prezzo alto e multiplayer introducono un rischio da monitorare, ma non dominante.

# Argomento 5 - Ragionamento logico e Knowledge Base Prolog

## Sommario

La Knowledge Base Prolog rappresenta il regolamento interno del publisher. Il suo compito non è ripetere la classificazione, ma applicare regole decisionali deterministiche combinando dati della proposta di gioco, cluster predetto dal modello supervisionato e rischio stimato dalla rete bayesiana.

## Strumenti utilizzati

La KB è implementata con SWI-Prolog e integrata in Python tramite `pyswip`. Le regole principali sono mantenute in `kb/publisher_rules.pl`, mentre i fatti sono generati automaticamente in `kb/generated_facts.pl` dallo script `src/prolog_reasoning.py`.

## Decisioni di Progetto

I fatti principali sono:

```prolog
gioco(Nome, Genere, Prezzo, Ore, Lingue, Multiplayer, Budget, ReviewScoreStimato).
predizione_commerciale(Nome, Cluster).
rischio_bayesiano(Nome, Rischio).
```

La separazione tra regole e fatti è una scelta importante: le regole rappresentano conoscenza stabile del publisher, mentre i fatti rappresentano istanze specifiche di giochi da valutare. In questo modo Prolog resta responsabile del ragionamento finale e Python si limita a generare i dati.

Le regole implementano:

- supporto globale: almeno 3 lingue;
- supporto premium: almeno 5 lingue;
- compatibilità del budget standard e ridotto;
- prezzo alto o basso;
- review score basso o buono;
- coerenza tra genere e durata;
- rischio accettabile o critico;
- violazioni bloccanti;
- approvazione, revisione e rifiuto.

La KB non viene usata come semplice archivio di fatti. Il verdetto finale deriva dalla composizione di più regole che integrano cluster, rischio bayesiano e caratteristiche del gioco, sfruttando il meccanismo inferenziale di Prolog.

### Risultati dimostrativi

| Gioco | Verdetto | Motivi bloccanti |
|---|---|---|
| `hades_like` | approvato |  |
| `short_puzzle_deluxe` | revisione |  |
| `underlocalized_action` | rifiutato | localizzazione insufficiente |
| `overpriced_multiplayer_rpg` | rifiutato | incoerenza genere; rischio critico |

I quattro casi sono stati scelti per coprire comportamenti diversi: approvazione diretta di un gioco coerente e promettente, richiesta di revisione per un gioco intermedio, rifiuto per vincolo commerciale non soddisfatto e rifiuto per combinazione di incoerenza e rischio.

## Valutazione

La KB è modulare: nuovi vincoli del publisher possono essere aggiunti introducendo nuove clausole, senza modificare clustering, classificazione o rete bayesiana. Il ragionamento Prolog usa risoluzione SLD su clausole di Horn e, nel progetto, il numero di fatti e regole è contenuto, quindi il costo delle query resta basso.

# Conclusioni

Il progetto realizza un sistema integrato di supporto alle decisioni. Il clustering produce profili commerciali; la classificazione supervisionata riconosce il profilo assegnato dalla fase non supervisionata; la rete bayesiana ragiona su scenari incerti; la KB Prolog applica vincoli aziendali espliciti per produrre un verdetto.

La valutazione supervisionata usa cross-validation ripetuta e metriche macro. La parte Prolog evita una KB usata come semplice database, perché il ragionamento finale dipende da regole multicriterio e da vincoli di coerenza. Il modello candidato per la predizione del cluster è Random Forest con SMOTE, che raggiunge il miglior F1 macro.

Le alternative più complesse, come modelli neurali, NLP sulle recensioni o recommender systems, sono state volutamente escluse. L’obiettivo non era massimizzare la complessità tecnica, ma costruire una pipeline coerente con i temi del corso e con il vincolo temporale del progetto. La scelta di pochi metadati commerciali, modelli interpretabili e regole esplicite rende il sistema più facile da motivare, valutare ed estendere.

# Sviluppi futuri

Possibili estensioni:

- generare i fatti Prolog direttamente da una proposta inserita dall’utente;
- addestrare una variante strettamente pre-lancio senza `Review_Count` e `Review_Score_Pct`;
- collegare automaticamente le probabilità bayesiane a livelli di rischio basso, medio e alto;
- aggiungere una piccola interfaccia per simulare scenari di investimento;
- testare la stabilità dei cluster su campioni diversi;
- introdurre ulteriori regole commerciali, ad esempio vincoli di piattaforma o requisiti per festival indie;
- esportare la relazione finale in formato PDF.

# Riferimenti Bibliografici

[1] Kaggle, “Steam Games Dataset”, fronkongames.

[2] pandas documentation.

[3] NumPy documentation.

[4] scikit-learn documentation.

[5] imbalanced-learn documentation.

[6] pgmpy documentation.

[7] SWI-Prolog documentation.

[8] pyswip documentation.

[9] Cartella `results/` con output sperimentali del progetto.
