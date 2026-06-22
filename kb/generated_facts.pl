% File generato da src/prolog_reasoning.py.

soglia_budget_standard(150000).
soglia_budget_ridotta(75000).
soglia_lingue_globale(3).
soglia_lingue_premium(5).
soglia_prezzo_alto(30).
soglia_prezzo_basso(10).
soglia_review_bassa(60).
soglia_review_buona(75).

gioco(hades_like, rpg, 24.99, 30, 8, 0, 120000, 88).
gioco(short_puzzle_deluxe, puzzle, 34.99, 4, 6, 0, 60000, 72).
gioco(underlocalized_action, action, 19.99, 9, 1, 1, 90000, 81).
gioco(overpriced_multiplayer_rpg, rpg, 49.99, 8, 5, 1, 180000, 55).

predizione_commerciale(hades_like, cluster_0).
predizione_commerciale(short_puzzle_deluxe, cluster_2).
predizione_commerciale(underlocalized_action, cluster_0).
predizione_commerciale(overpriced_multiplayer_rpg, cluster_1).

rischio_bayesiano(hades_like, basso).
rischio_bayesiano(short_puzzle_deluxe, medio).
rischio_bayesiano(underlocalized_action, basso).
rischio_bayesiano(overpriced_multiplayer_rpg, alto).
