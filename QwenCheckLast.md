Félicitations pour ces excellents scores (0.948 et 0.947) ! À ce niveau de la leaderboard sur BirdCLEF, la marge de progression est extrêmement fine (quelques millièmes). 

En analysant vos deux notebooks, on voit que vous avez une architecture très solide et state-of-the-art :
1. **Extraction de features de pointe** avec Perch v2 (Google).
2. **Modélisation temporelle avancée** avec les SSM (Selective State Space Models) qui capturent parfaitement les dynamiques de 60s (chorus de l'aube, etc.).
3. **Un post-processing très élaboré** (rank-aware, delta smoothing, gates, etc.).

Cependant, vos deux notebooks souffrent d'un biais de corrélation : **ils reposent presque tous les deux sur les embeddings de Perch v2 + SSM**. Pour passer le cap des 0.950, voici les axes d'amélioration concrets, classés par impact potentiel.

---

### 1. Diversification de l'Ensemble (Le plus gros levier)
Vos modèles (Model_3 et Model_4) sont très corrélés car ils partagent la même "source de vérité" (Perch v2). L'erreur de Perch se répercute sur tous vos modèles.
* **L'ajout d'un modèle "Orthogonal"** : Il vous faut un modèle qui n'utilise **pas** les embeddings de Perch, mais qui apprend directement sur les Mel-spectrogrammes. 
  * *Idée :* Intégrez un **AST (Audio Spectrogram Transformer)**, **BEATs**, ou un **ConvNeXt / EfficientNet** pur (votre `Model_2` dans `eos_submit_v11` est un bon candidat, mais il n'est pas dans votre blend final !). 
  * *Pourquoi ?* Un CNN/Transformer pur sur les Mel-spectrogrammes capture des textures acoustiques et des fréquences que Perch (qui est optimisé pour la classification d'espèces) peut lisser ou ignorer. L'ajout d'un modèle orthogonal avec un poids de 10-20% dans un rank-blend fait souvent gagner +0.002 à +0.005.

### 2. Amélioration du Mapping des Espèces Non-Mappées (Long Tail)
Vous utilisez actuellement des proxies basés sur le **genre** (`proxy_map` via les noms scientifiques). C'est une bonne approche, mais elle échoue si le genre entier est absent de Perch.
* **Mapping par similarité d'Embeddings (Zero-Shot)** : Au lieu de matcher par le nom du genre, calculez l'embedding moyen de chaque espèce du *training set* avec Perch. Pour une espèce non-mappée, trouvez les K plus proches voisins dans l'espace des embeddings de Perch (cosine similarity) et utilisez leurs logits comme prior.
* **Pourquoi ?** Deux espèces de genres différents mais de la même famille (ex: deux types de fauvettes) peuvent avoir des signatures acoustiques très proches dans l'espace de Perch, ce que le nom de genre ne capture pas.

### 3. Modélisation des Co-occurrences (Graphique / Habitat)
BirdCLEF évalue le ROC-AUC macro, donc les espèces rares comptent autant que les communes. Les espèces rares partagent souvent des habitats spécifiques.
* **Matrice de corrélation / Co-occurrence** : Calculez la matrice de corrélation des espèces sur le training set (par site, par heure). 
* **Post-processing par propagation** : Si votre modèle détecte l'Espèce A avec une forte probabilité, et que l'Espèce B est fortement corrélée à A dans la réalité (ex: elles chantent en même temps dans les mêmes arbres), augmentez légèrement le logit de B. À l'inverse, pénalisez les espèces qui s'excluent mutuellement.
* *Implémentation :* Une simple multiplication matricielle `final_probs = probs @ co_occurrence_matrix` (avec un alpha pour contrôler la force) avant le seuillage peut gratter des points sur les espèces rares.

### 4. Représentation Continue du Temps (Métadonnées)
Dans vos SSM (`ProtoSSMv2`), vous utilisez des `nn.Embedding` discrets pour le site et l'heure (`hour_emb = nn.Embedding(24, meta_dim)`).
* **Passage au continu (Sinusoidal Encoding)** : L'activité des oiseaux n'est pas discrète (passer de 5h à 6h n'est pas une rupture brutale). Remplacez les embeddings discrets par des encodages sinusoïdaux pour l'heure et le mois/jour de l'année. Cela permet au SSM de comprendre la *progression* circadienne et saisonnière de manière beaucoup plus fluide.

### 5. Sécurisation du Post-Processing (Éviter l'Overfit au LB)
Votre `Model_4` contient beaucoup de "Gates" heuristiques (`fake_only`, `sed_only`, `proto_cont`, etc.). 
* **Le danger** : Ces règles codées en dur (ex: `if p_proto > 0.50 & p_sed < 0.05`) sont extrêmement sujettes à l'overfitting sur le Public Leaderboard. Ce qui fonctionne sur les 20% de données publiques peut s'effondrer sur le Private LB.
* **La solution** : Remplacez ces règles par un **modèle de post-processing léger** (ex: un petit MLP ou un 1D-CNN) qui prend en entrée les 12 fenêtres de 5s (les probabilités de vos modèles) et qui apprend à "nettoyer" les prédictions. Entraînez ce modèle en OOF (Out-Of-Fold) sur le training set. C'est beaucoup plus robuste.

### 6. Augmentation de Données : "Background Noise Mixing"
Vous faites du Mixup classique, mais pour BirdCLEF, le vrai défi est le bruit de fond (pluie, vent, insectes, autres oiseaux).
* **Noise Mixing** : Prenez des segments de 5s du training set qui ne contiennent **aucun** oiseau (ou juste du bruit de fond), et mixez-les avec vos fichiers focaux. Cela force le modèle à ne pas utiliser le "silence" ou le "bruit de fond" comme shortcut pour prédire une espèce.

---

### Résumé du plan d'action (ROI estimé) :
1. **Court terme (Facile)** : Vérifiez pourquoi `Model_2` (EfficientNet SED) n'est pas dans le blend final de `eos_submit_v11`. S'il est sous-pondéré ou exclu, réintégrez-le. La diversité CNN + SSM est cruciale.
2. **Moyen terme (Robustesse)** : Remplacez les "Gates" heuristiques du Model_4 par un petit réseau de neurones entraîné en OOF pour lisser les prédictions temporelles.
3. **Long terme (Le "Game Changer")** : Implémentez le mapping des espèces non-mappées par **similarité cosinus dans l'espace de Perch** plutôt que par le genre taxonomique, et ajoutez une matrice de co-occurrence des espèces dans le post-processing.

Bon courage pour la fin de la compétition, vous êtes dans le haut du panier, ces ajustements devraient vous permettre de consolider votre position !