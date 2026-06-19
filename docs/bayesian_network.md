# Ragionamento probabilistico e rete bayesiana

## Obiettivo

La rete bayesiana serve a modellare l'incertezza commerciale del publisher. A differenza del classificatore supervisionato, che restituisce una predizione del cluster, la rete permette di interrogare probabilisticamente scenari incompleti.

Esempi di domande:

- quale profilo commerciale e piu probabile dato un prezzo alto e un certo genere?
- come cambia il rischio di recensioni basse se il gioco e multiplayer?
- quali relazioni emergono tra prezzo, durata, recensioni e cluster?

## Dataset usato

La rete usa:

- `data/processed/steam_games_discretized_clustered.csv`.

Si usa il dataset discretizzato per evitare problemi tipici delle reti bayesiane su valori continui, dove le CPD diventano troppo sparse o ingestibili.

## Variabili

Variabili incluse:

- `Primary_Genre`;
- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
- `Playtime_Hours`;
- `Languages_Count`;
- `Multiplayer`;
- `Cluster_Label`.

`Cluster_Label` collega la rete bayesiana alla fase di apprendimento automatico precedente.

## Apprendimento della struttura

Lo script `src/bayesian_network.py` usa:

- HillClimbSearch;
- score BIC;
- vincolo `max_indegree = 3`.

Il vincolo sul numero massimo di genitori limita la complessita della rete e rende le CPD piu gestibili.

## Apprendimento dei parametri

I parametri vengono appresi tramite Maximum Likelihood Estimator. Se durante l'esecuzione emergessero problemi legati a stati non osservati, si potra passare a un estimatore bayesiano con smoothing.

## Query previste

Lo script salva alcune query iniziali in:

- `results/bayesian_queries.csv`.

Le query includono:

- distribuzione di `Cluster_Label` dato genere frequente e prezzo alto;
- distribuzione di `Cluster_Label` dato prezzo basso e playtime lungo;
- distribuzione di `Review_Score_Pct` dato prezzo alto e multiplayer.

Gli archi appresi vengono salvati in:

- `results/bayesian_edges.csv`.

Questi output saranno commentati nella relazione, evitando di limitarsi alla sola visualizzazione della rete.

## Risultati ottenuti

La struttura appresa contiene i seguenti archi:

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

Gli archi piu rilevanti per il problema decisionale sono:

- `Review_Count -> Cluster_Label`, perche collega popolarita osservata e profilo commerciale;
- `Cluster_Label -> Review_Score_Pct`, perche il profilo commerciale influenza la ricezione utente;
- `Playtime_Hours -> Review_Count`, perche la durata risulta collegata al volume di attenzione ricevuta;
- `Cluster_Label -> Price`, utile per discutere la coerenza tra profilo commerciale e prezzo.

## Query eseguite

Gli stati discreti sono interpretati come:

- `0`: valore basso;
- `1`: valore medio;
- `2`: valore alto.

Per `Cluster_Label`, l'interpretazione segue il clustering:

- `0`: ricezione molto positiva;
- `1`: profilo a maggiore rischio commerciale;
- `2`: ricezione intermedia.

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

La terza query e particolarmente utile per il publisher: dato uno scenario con prezzo alto e multiplayer, la probabilita di review score basso e circa 36.61%, superiore alla probabilita di review score alto. Questo segnala un rischio di pricing o di aspettative utente da considerare nella decisione finale.

## Limite osservato

Le query usano stati numerici generati dalla discretizzazione. Nella relazione finale sara quindi necessario spiegare la mappatura basso/medio/alto e collegarla ai bin prodotti dal preprocessing.
