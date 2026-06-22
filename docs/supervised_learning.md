# Apprendimento supervisionato

## Obiettivo

La fase supervisionata usa `Cluster_Label`, ottenuto dal clustering KMeans, come variabile target. Il modello deve prevedere il profilo commerciale di un nuovo gioco prima della pubblicazione.

In questo modo il clustering non resta un esercizio isolato: produce le classi operative usate dal sistema decisionale.

## Feature e target

Dataset di partenza:

- `data/processed/steam_games_normalized_clustered.csv`.

Target:

- `Cluster_Label`.

Feature usate:

- `Primary_Genre`;
- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
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

La classe meno rappresentata e il cluster 1, corrispondente al profilo a maggiore rischio commerciale. Il confronto con SMOTE e quindi utile per verificare se il bilanciamento migliora la stabilita del sistema proprio sulla classe piu critica.

Risultati mediati su 5 fold ripetuti 3 volte:

| Modello | SMOTE | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---:|---:|---:|---:|
| Random Forest | No | 0.9929 +/- 0.0018 | 0.9930 +/- 0.0018 | 0.9945 +/- 0.0014 | 0.9937 +/- 0.0015 |
| Random Forest | Si | 0.9933 +/- 0.0020 | 0.9933 +/- 0.0023 | 0.9950 +/- 0.0015 | 0.9941 +/- 0.0019 |
| SVM | No | 0.9878 +/- 0.0037 | 0.9879 +/- 0.0046 | 0.9869 +/- 0.0042 | 0.9874 +/- 0.0042 |
| SVM | Si | 0.9804 +/- 0.0038 | 0.9735 +/- 0.0054 | 0.9837 +/- 0.0037 | 0.9784 +/- 0.0045 |

La Random Forest con SMOTE e il modello migliore sul F1 macro. Il miglioramento rispetto alla versione senza SMOTE e contenuto ma coerente sulle metriche macro, segnalando un lieve beneficio nel trattare la classe minoritaria.

Per SVM, invece, SMOTE aumenta la recall macro ma riduce precision macro e F1 macro. In questo caso la versione senza SMOTE risulta preferibile.

Migliori parametri trovati:

| Modello | SMOTE | Parametri migliori |
|---|---|---|
| Random Forest | No | `max_depth=20`, `min_samples_leaf=1`, `n_estimators=200` |
| Random Forest | Si | `max_depth=10`, `min_samples_leaf=1`, `n_estimators=200` |
| SVM | No | `C=10`, `gamma=scale`, `kernel=linear` |
| SVM | Si | `C=10`, `gamma=scale`, `kernel=linear` |

Conclusione operativa: per predire il profilo commerciale di un nuovo gioco, il candidato principale e Random Forest con SMOTE. Il risultato verra usato come input della KB Prolog nella fase decisionale finale.
