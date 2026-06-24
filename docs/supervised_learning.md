# Apprendimento supervisionato

## Obiettivo

La fase supervisionata usa `Cluster_Label`, ottenuto dal clustering KMeans, come variabile target. Il modello deve riconoscere il profilo commerciale di un gioco in scenario pre-lancio, usando solo metadati disponibili o stimabili prima della pubblicazione.

Per questo motivo `Review_Count` e `Review_Score_Pct` non vengono usati come feature supervisionate: restano utili per costruire e interpretare i cluster, ma sarebbero segnali post-pubblicazione non disponibili per una nuova proposta.

In questo modo il clustering non resta un esercizio isolato: produce le classi operative usate dal sistema decisionale.

## Feature e target

Dataset di partenza:

- `data/processed/steam_games_normalized_clustered.csv`.

Target:

- `Cluster_Label`.

Feature usate:

- `Primary_Genre`;
- `Price`;
- `Playtime_Hours`;
- `Languages_Count`;
- `Multiplayer`.

`Primary_Genre` viene codificato tramite one-hot encoding dentro la pipeline, per evitare trasformazioni manuali non tracciate.

## Modelli confrontati

Vengono confrontati:

- Random Forest;
- SVM.

La Random Forest e utile come modello robusto su feature eterogenee; la SVM consente di verificare il comportamento di una macchina a kernel su una rappresentazione tabulare normalizzata.

## Valutazione

La valutazione usa `RepeatedStratifiedKFold` con:

- 5 fold;
- 3 ripetizioni;
- seed fisso.

Le metriche salvate sono:

- accuracy;
- precision macro;
- recall macro;
- F1 macro.

Per ogni metrica vengono riportate media e deviazione standard. Questa scelta evita di basare le conclusioni su un singolo split o su un singolo classification report.

## Gestione dello sbilanciamento

Il dataset clusterizzato non e perfettamente bilanciato. Per questo ogni modello viene valutato in due versioni:

- senza SMOTE;
- con SMOTE.

SMOTE viene applicato dentro la pipeline di cross-validation, quindi solo sui fold di training. Questo evita data leakage verso i fold di validazione.

## Output previsti

Lo script `src/supervised_learning.py` produce:

- `results/class_distribution.csv`;
- `results/supervised_metrics.csv`;
- `results/supervised_best_params.csv`.

## Risultati ottenuti

Distribuzione delle classi:

| Cluster | Esempi | Percentuale |
|---|---:|---:|
| 0 | 2437 | 48.74% |
| 1 | 757 | 15.14% |
| 2 | 1806 | 36.12% |

La classe meno rappresentata e il cluster 1, corrispondente al profilo a maggiore rischio commerciale. Il confronto con SMOTE resta utile per verificare se il bilanciamento migliora la stabilita del sistema proprio sulla classe piu critica.

Risultati mediati su 5 fold ripetuti 3 volte:

| Modello | SMOTE | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|
| Random Forest | No | 0.4947 +/- 0.0168 | 0.4316 +/- 0.0189 | 0.4139 +/- 0.0148 | 0.4153 +/- 0.0155 |
| Random Forest | Si | 0.4659 +/- 0.0149 | 0.4305 +/- 0.0122 | 0.4488 +/- 0.0126 | 0.4298 +/- 0.0125 |
| SVM | No | 0.4966 +/- 0.0104 | 0.4908 +/- 0.0658 | 0.3638 +/- 0.0093 | 0.3220 +/- 0.0134 |
| SVM | Si | 0.3925 +/- 0.0148 | 0.3975 +/- 0.0151 | 0.4184 +/- 0.0139 | 0.3695 +/- 0.0136 |

Migliori parametri trovati:

| Modello | SMOTE | Parametri migliori |
|---|---|---|
| Random Forest | No | `max_depth=20`, `min_samples_leaf=1`, `n_estimators=100` |
| Random Forest | Si | `max_depth=20`, `min_samples_leaf=3`, `n_estimators=200` |
| SVM | No | `C=10`, `gamma=scale`, `kernel=rbf` |
| SVM | Si | `C=10`, `gamma=scale`, `kernel=rbf` |

Conclusione operativa: nello scenario pre-lancio il candidato principale e Random Forest con SMOTE, perche ottiene il miglior F1 macro. Il risultato verra usato come input della KB Prolog nella fase decisionale finale.
