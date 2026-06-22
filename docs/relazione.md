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

### Scelta progettuale generale

La scelta principale del progetto e stata costruire un sistema integrato invece di limitarsi a un singolo modello predittivo. Una semplice classificazione avrebbe permesso di assegnare un'etichetta a un gioco, ma non avrebbe mostrato come tale risultato possa essere usato in un processo decisionale piu ampio. Per questo motivo la pipeline collega quattro livelli:

- il clustering scopre profili commerciali non definiti a priori;
- la classificazione supervisionata impara a riconoscere tali profili;
- la rete bayesiana valuta scenari incerti e relazioni probabilistiche;
- Prolog applica regole aziendali esplicite e produce il verdetto finale.

Sono state escluse impostazioni alternative come sistemi di raccomandazione, analisi del testo delle recensioni o classificazione di immagini promozionali. Queste alternative sarebbero state interessanti, ma avrebbero spostato il progetto verso temi meno centrali per ICon e avrebbero richiesto componenti NLP o computer vision non necessarie per dimostrare l'integrazione tra apprendimento e ragionamento.

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

Sono stati considerati anche dataset piu piccoli e gia puliti, come raccolte limitate ai giochi piu popolari. Tali dataset avrebbero semplificato il preprocessing, ma avrebbero ridotto la varieta dei profili commerciali e reso meno interessante la fase di clustering. Il dataset scelto richiede invece una fase di pulizia piu attenta, ma permette di lavorare su un mercato piu ampio e realistico.

Non e stato usato l'intero dataset per due motivi. Il primo e computazionale: la cross-validation ripetuta, SMOTE e la rete bayesiana aumentano il costo dell'esperimento. Il secondo e metodologico: molti giochi nel dataset hanno pochissime recensioni o informazioni di mercato quasi nulle; includerli avrebbe introdotto rumore e reso meno interpretabili i cluster.

### Correzione dell'intestazione

Durante la prima lettura del CSV e emersa un'anomalia: nell'intestazione compare il campo `DiscountDLC count`, mentre nelle righe dati `Discount` e `DLC count` sono due colonne separate. Se il file viene letto senza correzione, tutte le colonne successive risultano sfalsate.

La correzione viene gestita direttamente nel codice di preprocessing:

- si legge la riga di intestazione;
- quando viene trovato `DiscountDLC count`, viene sostituito con due colonne;
- le colonne successive tornano allineate;
- il file CSV originale non viene modificato manualmente.

Questa scelta rende la pipeline riproducibile: chi esegue il progetto puo partire dal dataset originale e ottenere lo stesso dataset processato.

L'implementazione si trova in `src/preprocessing.py`. La funzione `build_corrected_header` legge l'intestazione originale, sostituisce il campo fuso con due nomi distinti e passa questa lista a `pandas.read_csv`. In questo modo la correzione non dipende da modifiche manuali al file e puo essere rieseguita in ogni ambiente.

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

In una prima fase erano state valutate anche feature come `Required age`, numero di piattaforme supportate, raccomandazioni e stima dei possessori. Sono state rimosse per evitare di introdurre variabili difficili da motivare nel dominio decisionale scelto. Ad esempio, l'eta richiesta puo influenzare il pubblico potenziale, ma nel progetto non veniva poi usata in modo esplicito dalla KB; mantenerla avrebbe reso la documentazione meno coerente. Allo stesso modo, raccomandazioni e possessori stimati sono proxy di popolarita parzialmente sovrapposti a `Review_Count`.

La selezione finale privilegia quindi poche feature ma direttamente giustificabili:

- prezzo, per il posizionamento commerciale;
- recensioni, per volume e ricezione;
- playtime, per stimare l'offerta di contenuto;
- genere, per la coerenza commerciale;
- lingue e multiplayer, per vincoli e rischio di publishing.

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

La normalizzazione e implementata con `MinMaxScaler`, mentre la discretizzazione usa `KBinsDiscretizer` con strategia `quantile`. La strategia quantile e stata preferita a una suddivisione uniforme perche alcune variabili, come il numero di recensioni, hanno distribuzioni molto sbilanciate: pochi giochi accumulano moltissime recensioni, mentre la maggioranza resta su valori piu bassi. Con i quantili, i bin risultano piu popolati e quindi piu utili per stimare probabilita nella rete bayesiana.

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

Non sono state incluse feature categoriche nel clustering per mantenere lo spazio geometrico piu semplice e interpretabile. Inserire direttamente il genere avrebbe richiesto codifica one-hot, aumentando la dimensionalita e rendendo meno chiara l'interpretazione dei centroidi. Il genere viene comunque recuperato nella sintesi dei cluster, osservando il genere piu frequente per ciascun gruppo.

### Perche KMeans e non altri algoritmi

KMeans e stato scelto per tre ragioni:

- produce centroidi facilmente interpretabili tramite medie delle feature;
- e adatto a dati numerici normalizzati;
- consente di usare l'Elbow Method per motivare il numero di cluster.

Sono state considerate alternative come clustering gerarchico e DBSCAN. Il clustering gerarchico avrebbe prodotto una struttura piu ricca, ma meno immediata da riutilizzare come target supervisionato in una pipeline compatta. DBSCAN avrebbe potuto individuare outlier, ma richiede una scelta delicata di `eps` e `min_samples`; inoltre, su dati di mercato con densita molto variabile, rischia di produrre molti punti rumore o cluster difficili da usare come classi. Per il nostro obiettivo, cioe creare profili commerciali latenti riutilizzabili in classificazione e Prolog, KMeans e risultato piu lineare e documentabile.

Non e stata usata PCA come fase principale perche il numero di feature del clustering e gia ridotto. Una riduzione ulteriore avrebbe reso meno immediata l'interpretazione business dei cluster, che invece e essenziale per motivare il target supervisionato.

### Scelta del numero di cluster

Il numero di cluster e stato valutato tramite Elbow Method per valori di `K` da 1 a 10. I risultati sono salvati in `results/kmeans_elbow.csv` e il grafico in `results/kmeans_elbow.png`.

Il valore scelto e `K = 3`. La scelta e coerente sia con la curva del gomito sia con l'obiettivo interpretativo del progetto: distinguere giochi con ricezione molto positiva, giochi intermedi e giochi a rischio commerciale piu alto.

Parametri principali:

- `K = 3`;
- `n_init = 10`;
- `max_iter = 300`;
- seed fisso per riproducibilita.

Nel codice, la fase e implementata in `src/clustering.py`. Lo script carica dataset pulito, normalizzato e discretizzato; esegue l'Elbow Method sul dataset normalizzato; applica KMeans con `SELECTED_K = 3`; infine copia la colonna `Cluster_Label` in tutti i dataset processati. Questa scelta evita disallineamenti tra dataset usati da modelli diversi.

### Risultati del clustering

La tabella dei centroidi interpretati e:

| Cluster | Giochi | Prezzo medio | Review score medio | Recensioni mediane | Playtime medio | Interpretazione |
|---|---:|---:|---:|---:|---:|---|
| 0 | 2437 | 6.620 | 89.993 | 639 | 18.380 | ricezione molto positiva |
| 1 | 757 | 4.215 | 48.602 | 143 | 6.374 | rischio commerciale piu alto |
| 2 | 1806 | 5.243 | 73.197 | 303 | 10.170 | ricezione intermedia |

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

La scelta di confrontare questi due modelli e legata alla loro natura diversa. Random Forest e un modello ensemble robusto su feature eterogenee e capace di modellare relazioni non lineari senza richiedere forti assunzioni sulla distribuzione dei dati. SVM, invece, rappresenta una macchina a kernel/margine che permette di verificare se una separazione geometrica nello spazio normalizzato e sufficiente a distinguere i cluster.

Non sono stati scelti modelli neurali perche il dataset finale e tabulare, di dimensione gestibile e con poche feature. Una rete neurale avrebbe introdotto piu iperparametri e minore interpretabilita senza un chiaro vantaggio per gli obiettivi del progetto. Non e stata scelta nemmeno una regressione logistica come modello principale perche i cluster non sono necessariamente separabili linearmente e il confronto con SVM lineare copre gia una parte di questa ipotesi.

Per Random Forest sono stati ricercati:

- `n_estimators`: 100, 200;
- `max_depth`: `None`, 10, 20;
- `min_samples_leaf`: 1, 3.

Per SVM sono stati ricercati:

- `C`: 0.1, 1, 10;
- `kernel`: `rbf`, `linear`;
- `gamma`: `scale`.

La griglia e stata mantenuta compatta per rispettare i tempi del progetto, ma sufficiente a confrontare modelli con capacita diverse.

L'ottimizzazione viene implementata con `GridSearchCV`. Per ogni modello vengono testate le combinazioni della griglia e la configurazione finale viene scelta usando F1 macro come metrica di refit. Questa scelta e coerente con il problema: non interessa massimizzare soltanto l'accuracy complessiva, ma mantenere buone prestazioni anche sulla classe minoritaria, che corrisponde al profilo a maggiore rischio commerciale.

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
| 0 | 2437 | 48.74% |
| 1 | 757 | 15.14% |
| 2 | 1806 | 36.12% |

Per questo motivo ogni modello viene valutato anche con SMOTE. SMOTE e inserito dentro la pipeline di cross-validation: viene applicato solo al training fold, evitando data leakage.

SMOTE non viene applicato prima della cross-validation perche cio introdurrebbe informazioni sintetiche derivate anche dai dati che dovrebbero restare nel fold di validazione. Nel codice, l'uso di `imblearn.pipeline.Pipeline` garantisce che oversampling, preprocessing e training siano ripetuti correttamente all'interno di ogni split.

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

La Random Forest con SMOTE ottiene il miglior F1 macro, pari a 0.9941 +/- 0.0019. Il miglioramento rispetto alla versione senza SMOTE e contenuto ma coerente sulle metriche macro, segnalando un lieve beneficio nel trattare la classe minoritaria.

Per SVM, invece, SMOTE non porta un miglioramento complessivo: la recall macro aumenta, ma precision macro e F1 macro diminuiscono. La versione SVM senza SMOTE e quindi preferibile alla versione con SMOTE, ma resta inferiore alla Random Forest.

Il modello candidato per l'integrazione nel sistema decisionale e Random Forest con SMOTE.

Il risultato va interpretato con cautela: le metriche molto alte dipendono anche dal fatto che il target supervisionato e stato generato dal clustering sulle stesse feature commerciali. Questo non rende inutile la fase supervisionata, perche il suo scopo non e scoprire una ground truth esterna, ma imparare a replicare il profilo commerciale latente per nuove proposte. Per questo motivo la valutazione viene presentata come stabilita della pipeline, non come prova di predizione di successo reale sul mercato.

## Capitolo 4 - Ragionamento probabilistico e rete bayesiana

### Obiettivo

La rete bayesiana viene usata per ragionare in condizioni di incertezza. Il classificatore supervisionato assegna un profilo commerciale, mentre la rete bayesiana permette di interrogare scenari parziali, ad esempio:

- cosa succede se il prezzo e alto?
- il multiplayer aumenta il rischio di recensioni basse?
- quale cluster e piu probabile dato un certo genere e un certo prezzo?

La rete bayesiana non sostituisce il classificatore supervisionato. Il suo ruolo e diverso: mentre Random Forest e SVM restituiscono una predizione, la rete bayesiana permette di ragionare su dipendenze probabilistiche e su variabili non completamente osservate. Questo e piu vicino a una situazione decisionale reale, in cui il publisher conosce alcune caratteristiche del gioco ma non conosce ancora la ricezione finale degli utenti.

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

Non e stata usata una rete bayesiana continua per due motivi. Il primo e pratico: molte implementazioni discrete, come quelle usate in `pgmpy`, richiedono stati enumerabili per stimare CPD gestibili. Il secondo e interpretativo: stati come prezzo basso, medio e alto sono piu facilmente traducibili in regole decisionali e in commenti per il publisher rispetto a valori continui molto frammentati.

### Apprendimento della struttura e dei parametri

La struttura viene appresa tramite HillClimbSearch con score BIC. E stato imposto `max_indegree = 3`, cosi ogni nodo puo avere al massimo tre genitori. La scelta limita la complessita della rete e rende piu leggibili le dipendenze apprese.

I parametri vengono appresi con l'estimatore discreto compatibile con la versione installata di `pgmpy`.

Sono state considerate due alternative: definire manualmente la struttura della rete oppure apprenderla automaticamente. La struttura manuale avrebbe permesso di imporre relazioni intuitive, ad esempio prezzo e durata verso review score. Tuttavia avrebbe rischiato di riflettere solo assunzioni progettuali. Lo structure learning consente invece di ottenere una rete guidata dai dati. Per evitare una rete eccessivamente complessa, e stato introdotto il vincolo `max_indegree = 3`.

Nel codice, questa parte e implementata in `src/bayesian_network.py`. Lo script carica il dataset discretizzato e clusterizzato, apprende la struttura con HillClimbSearch, stima le CPD e salva sia gli archi appresi sia le query in formato CSV. Sono stati gestiti anche cambiamenti di API tra versioni di `pgmpy`, usando `DiscreteBayesianNetwork` e `DiscreteMLE` quando disponibili.

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

Alcuni archi non vanno letti come causalita certa, ma come dipendenze probabilistiche apprese sui dati. Ad esempio, `Cluster_Label -> Price` non significa che il cluster causi il prezzo nel mondo reale; significa che, nella distribuzione osservata, prezzo e cluster risultano statisticamente collegati secondo la struttura appresa. Nella relazione vengono quindi usati come supporto interpretativo, non come dimostrazione causale.

### Inferenza

Risultati principali:

| Query | Stato | Probabilita |
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

La query sul prezzo alto e multiplayer e rilevante per il publisher: la probabilita di review score basso e circa 31.39%, mentre lo stato medio e il piu probabile. Questo risultato suggerisce che, in uno scenario con prezzo alto e componente multiplayer, esiste un rischio da monitorare, ma non dominante rispetto agli altri stati di ricezione.

## Capitolo 5 - Ragionamento logico e Knowledge Base Prolog

### Obiettivo

La KB Prolog rappresenta il regolamento interno del publisher. Il suo compito non e ripetere la classificazione, ma applicare regole decisionali deterministiche combinando:

- dati della proposta di gioco;
- cluster predetto dal modello supervisionato;
- rischio stimato dalla rete bayesiana.

La scelta di usare Prolog, invece di semplici `if` in Python, e motivata dalla necessita di rappresentare in modo dichiarativo la conoscenza del publisher. Le regole Prolog rendono esplicite le condizioni di approvazione, revisione e rifiuto, separandole dal codice procedurale. Questo permette di discutere la KB come componente autonoma del sistema, non come semplice post-processing numerico.

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

La separazione tra regole e fatti e una scelta implementativa importante:

- le regole rappresentano conoscenza stabile del publisher;
- i fatti rappresentano istanze specifiche di giochi da valutare;
- Python genera i fatti in ordine, evitando warning Prolog sui predicati non contigui;
- Prolog resta responsabile del ragionamento finale.

In una versione estesa, i fatti potrebbero essere generati direttamente dall'output del classificatore e dalle probabilita della rete bayesiana. Nel prototipo attuale sono presenti casi dimostrativi costruiti per verificare i diversi rami della KB.

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

Le soglie principali sono definite come fatti/regole Prolog: budget standard, budget ridotto, prezzo alto, prezzo basso, review score basso e review score buono. Questo rende le soglie modificabili senza cambiare la struttura delle regole.

Esempio di logica decisionale:

- un gioco in cluster di successo puo essere approvato se ha supporto globale, budget compatibile, coerenza genere-durata e rischio accettabile;
- un gioco intermedio puo essere approvato solo con supporto premium, budget ridotto, review score buono e nessuna violazione;
- un gioco con rischio critico, localizzazione insufficiente o incoerenza di genere viene rifiutato.

La regola di coerenza genere-durata e stata inserita per evitare che la KB si limiti a confrontare valori numerici isolati. Ad esempio, un RPG con durata molto bassa viene considerato incoerente rispetto alle aspettative commerciali del genere, mentre un puzzle game puo essere coerente anche con una durata piu breve. Questo tipo di vincolo rende il ragionamento piu vicino a una decisione editoriale reale.

### Perche la KB non e pattern matching

La KB non si limita a cercare fatti esplicitamente presenti. Il verdetto viene derivato da catene di regole. Ad esempio:

- `verdetto/2` dipende da `approva_finanziamento/1`, `richiede_revisione/1` e `rifiuta_finanziamento/1`;
- `approva_finanziamento/1` combina cluster, lingue, budget, durata e rischio;
- `violazione_bloccante/2` astrae cause diverse di rifiuto;
- `rischio_critico/1` puo derivare sia da rischio bayesiano alto sia dalla combinazione di prezzo alto e review score basso.

Questa struttura integra output numerici e probabilistici in regole aziendali interpretabili.

Un'alternativa sarebbe stata usare una piccola ontologia o una tabella di regole implementata in Python. Questa soluzione e stata scartata perche avrebbe rischiato di ridurre la KB a un database consultato tramite pattern matching. Prolog, invece, permette di esprimere relazioni derivate, negazione come fallimento e regole alternative per lo stesso predicato, rendendo piu evidente la componente di ragionamento logico.

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

Il modello candidato per la predizione del cluster e Random Forest con SMOTE, che raggiunge il miglior F1 macro. Il vantaggio rispetto alla versione senza SMOTE e contenuto, ma utile per migliorare il comportamento medio sulla classe minoritaria. La rete bayesiana fornisce invece un supporto probabilistico per valutare rischio di prezzo e ricezione utente. La KB conclude il processo trasformando questi risultati in una decisione interpretabile.

Le alternative piu complesse, come modelli neurali, NLP sulle recensioni o recommender systems, sono state volutamente escluse. L'obiettivo non era massimizzare la complessita tecnica, ma costruire una pipeline coerente con i temi del corso e con il vincolo temporale del progetto. La scelta di pochi metadati commerciali, modelli interpretabili e regole esplicite rende il sistema piu facile da motivare, valutare ed estendere.

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
