# Dataset e preprocessing

## Dataset scelto

Il progetto usa il dataset Kaggle "Steam Games Dataset" di fronkongames. La scelta e motivata da tre aspetti:

- contiene dati tabulari su videogiochi Steam, quindi evita NLP e riconoscimento di immagini;
- include variabili commerciali utili come prezzo, recensioni, generi, lingue, playtime e categorie;
- ha dimensione ampia, ma puo essere campionato in modo controllato per rispettare il vincolo di circa 25 ore.

## Correzione dell'intestazione

Nel file CSV originale l'intestazione contiene il campo `DiscountDLC count`, mentre nei dati `Discount` e `DLC count` sono due colonne distinte. Senza correzione, tutte le colonne successive risultano sfalsate.

La pipeline corregge il problema in lettura sostituendo `DiscountDLC count` con:

- `Discount`;
- `DLC count`.

Il CSV originale non viene modificato manualmente.

## Feature derivate

Dal dataset originale vengono costruite le seguenti feature di progetto:

- `Primary_Genre`: primo genere indicato in `Genres`;
- `Review_Count`: somma di recensioni positive e negative;
- `Review_Score_Pct`: percentuale di recensioni positive;
- `Playtime_Hours`: playtime medio convertito da minuti a ore;
- `Languages_Count`: numero di lingue supportate;
- `Multiplayer`: variabile booleana derivata dalle categorie Steam.

## Filtri applicati

Per evitare che il clustering sia dominato da giochi privi di segnali commerciali, vengono esclusi:

- giochi senza nome o genere principale;
- giochi con meno di 20 recensioni totali;
- giochi con prezzo fuori dall'intervallo 0-100;
- duplicati sul campo `AppID`.

Il dataset pulito viene poi campionato a 5000 righe con seed fisso. Questa scelta mantiene il progetto gestibile e rende riproducibili gli esperimenti.

## Output prodotti

La pipeline genera tre file:

- `data/processed/steam_games_clean.csv`: dataset pulito con feature derivate;
- `data/processed/steam_games_normalized.csv`: dataset normalizzato per clustering e modelli basati su distanza;
- `data/processed/steam_games_discretized.csv`: dataset discretizzato per la rete bayesiana.
