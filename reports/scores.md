Local CV

Sem sampling

1. LGBM simples, todas as variáveis, sem otimização, np.nan: 0.6395581804921483
2. LGBM simples, todas as variáveis, setando is_unbalance, np.nan: 0.6594159675776389
3. LGBM simples, todas as variáveis, setando is_unbalance, nulos como -999: 0.6608989474619081
5. LGBM otimizado, todas as variáveis, setando is_unbalance, {'num_leaves': 43, 'learning_rate': 0.028537628044991967, 'n_estimators': 397}: 0.6690070937237789

Random UnderSampling

1. LGBM simples, todas as variáveis, Random UnderSampling, {'sampling_strategy': 0.5199689404423158, 'is_unbalance': False}: 0.6689187007984112
1. LGBM otimizada, todas as variáveis, Random UnderSampling otimizada acima, {'num_leaves': 54, 'learning_rate': 0.07949316455644706, 'n_estimators': 485}: 0.8693638510639261

Near Miss 

1. LGBM simples, todas as variáveis, Near Miss v1, {'sampling_strategy': 0.37852656804590223, 'n_neighbors': 3}: 0.6516043302160928
2. LGBM simples, todas as variáveis, Near Miss v2, {'sampling_strategy': 0.2721875849730608, 'n_neighbors': 3}: 0.6435453345573452
3. LGBM simples, todas as variáveis, Near Miss v3, {'n_neighbors_ver3': 9, 'sampling_strategy': 0.5316695530547733, 'n_neighbors': 9}: 0.6550513056849
4. Todas as otimizações de LGBM com os parâmetros acima estão overfitados.

Tomek Links

1. LGBM simples, todas as variáveis, Tomek Links, setando is_unbalance: 0.6609584111039281
2. LGBM otimizado, todas as variáveis, Tomek Links setando, {'num_leaves': 58, 'learning_rate': 0.01873703328456237, 'n_estimators': 479, 'is_unbalance': True}: 0.668531756273046