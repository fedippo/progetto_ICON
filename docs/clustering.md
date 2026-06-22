# Apprendimento non supervisionato

## Obiettivo

Il clustering serve a trasformare il dataset Steam in profili commerciali latenti. Questi profili diventano poi la classe da predire nella fase supervisionata e un'informazione usata dalla KB Prolog.

Questa scelta evita di trattare il clustering come esercizio isolato: il risultato viene riutilizzato nelle fasi successive del sistema decisionale.

## Feature usate

Il KMeans viene applicato alle feature numeriche normalizzate:

- `Price`;
- `Review_Score_Pct`;
- `Review_Count`;
- `Playtime_Hours`.

La normalizzazione e necessaria perche KMeans usa distanze euclidee: senza riportare le feature sulla stessa scala, il numero di recensioni dominerebbe prezzo e durata.

## Scelta del numero di cluster

E stato calcolato l'Elbow Method per `K` da 1 a 10. I valori ottenuti sono salvati in:

- `results/kmeans_elbow.csv`.

Nel setup definitivo, con `matplotlib` installato, lo script genera anche:

- `results/kmeans_elbow.png`.

Per ora si mantiene `K = 3`, coerente con l'idea progettuale iniziale:

- giochi con ricezione molto positiva;
- giochi con ricezione intermedia;
- giochi a maggiore rischio commerciale.

## Risultati ottenuti

La tabella riassuntiva e salvata in:

- `results/cluster_summary.csv`.

Risultato attuale:

| Cluster | Giochi | Prezzo medio | Review score medio | Recensioni mediane | Playtime medio | Interpretazione provvisoria |
|---|---:|---:|---:|---:|---:|---|
| 0 | 2437 | 6.620 | 89.993 | 639 | 18.380 | ricezione molto positiva |
| 1 | 757 | 4.215 | 48.602 | 143 | 6.374 | rischio commerciale piu alto |
| 2 | 1806 | 5.243 | 73.197 | 303 | 10.170 | ricezione intermedia |

Le etichette semantiche saranno affinate dopo aver osservato anche distribuzioni, generi e risultati della classificazione.

Gli identificativi numerici dei cluster sono assegnati da KMeans e non hanno significato intrinseco. L'interpretazione usata nel progetto deriva dalle statistiche dei cluster e viene mantenuta coerente nelle fasi supervisionata, bayesiana e Prolog.

## Output prodotti

Lo script `src/clustering.py` genera:

- `data/processed/steam_games_clean_clustered.csv`;
- `data/processed/steam_games_normalized_clustered.csv`;
- `data/processed/steam_games_discretized_clustered.csv`;
- `results/kmeans_elbow.csv`;
- `results/cluster_summary.csv`.
