% File generato da src/prolog_reasoning.py.

soglia_lingue_globale(3).
soglia_lingue_premium(5).
soglia_prezzo_alto(30).
soglia_prezzo_basso(10).

gioco(stellar_strategy_master, strategy, 19.99, 50, 12, 0).
gioco(short_puzzle_deluxe, puzzle, 34.99, 4, 6, 0).
gioco(underlocalized_action, action, 19.99, 9, 1, 1).
gioco(overpriced_multiplayer_rpg, rpg, 49.99, 8, 5, 1).

predizione_commerciale(stellar_strategy_master, cluster_0).
predizione_commerciale(short_puzzle_deluxe, cluster_0).
predizione_commerciale(underlocalized_action, cluster_2).
predizione_commerciale(overpriced_multiplayer_rpg, cluster_2).

rischio_bayesiano(stellar_strategy_master, basso).
rischio_bayesiano(short_puzzle_deluxe, basso).
rischio_bayesiano(underlocalized_action, alto).
rischio_bayesiano(overpriced_multiplayer_rpg, alto).

probabilita_recensioni_basse(stellar_strategy_master, 0.000000).
probabilita_recensioni_basse(short_puzzle_deluxe, 0.000000).
probabilita_recensioni_basse(underlocalized_action, 0.572434).
probabilita_recensioni_basse(overpriced_multiplayer_rpg, 0.572434).
