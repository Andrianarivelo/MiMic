# Probabilité des vocalisations et comportements sociaux

Le script `scripts/social_call_probability.py` analyse les 13 comportements sociaux pendant la présence du partenaire (300 à 900 s). Les fichiers bruts et les métadonnées sont lus sans modification. La résidente et le partenaire sont distingués pour les comportements directionnels, mais le microphone mesure les appels de la dyade, sans identification du locuteur.

## Lancement

Double-cliquer sur `scripts/social_call_probability.cmd`, lancer le script dans un IDE, ou utiliser :

```powershell
python scripts/social_call_probability.py
python scripts/social_call_probability.py --figures-only
python scripts/social_call_probability.py --recompute
```

Dépendances : NumPy, pandas, SciPy, statsmodels, threadpoolctl, Matplotlib et les dépendances du module existant `social_vocal_correlations.py`. La configuration numérique est au début du script principal. Les couleurs, dimensions et dispositions sont au début de `social_call_probability_figures.py`.

## Mesures et inférence

- Pendant : probabilité d'au moins un début d'appel dans un intervalle complètement observé de 100 ms, occupé au moins à moitié par le comportement.
- Autour : probabilité d'au moins un appel dans [-1,+1) s autour du début d'un épisode. Les débuts sont quantifiés à 100 ms. Mesures avant et après séparées, ainsi que courbes descriptives à ±5 s.
- Taux exact : appels par minute de temps comportemental réellement occupé, calculés à la résolution des images, sans ajouter les interruptions fusionnées.
- Épisodes : interruptions observées fusionnées jusqu'à 0,2 s, durée active minimale 0,2 s. Débuts censurés exclus. Les fenêtres doivent rester entièrement observées et dans un même bloc local.
- Une session doit atteindre 95 % de couverture commune. L'analyse principale conserve 23 sessions (12 WT, 11 HET). Les comportements absents ou insuffisamment observés restent indéfinis.
- Chaque session a le même poids dans son génotype. Le résultat combiné standardise à 50 % WT et 50 % HET. Les 4 000 bootstraps rééchantillonnent les sessions, pas les épisodes.
- Le niveau attendu utilise tous les décalages cycliques possibles, y compris zéro, des appels dans les segments observés de blocs locaux de 60 s. Le test bilatéral utilise 19 999 combinaisons de décalages. Cette référence suppose une stationnarité locale et conserve la structure uniquement à l'intérieur des segments.
- BH, BY et Holm sont exportés pour les 26 tests principaux. Les analyses de génotype, avant/après et acteur ont leurs propres familles secondaires. Les 52 tests de sensibilité aux blocs de 30/120 s forment une famille secondaire supplémentaire.

Les comportements se chevauchent. Une probabilité élevée ne prouve ni l'identité du locuteur ni que le comportement dépasse statistiquement tous les autres. Les partenaires réutilisés peuvent induire une dépendance résiduelle, évaluée par une sensibilité descriptive.

## Résultats et contrôle

Les sorties sont dans `lgdel_usv_analysis/social_call_probability/` : rapport français, statistiques complètes, probabilités par session, dénominateurs, courbes, contrôles des sources, sensibilités et provenance. Six figures sont disponibles chacune en PNG, PDF et SVG, sur fond blanc.

Les empreintes des fonctions numériques, métadonnées, configurations et fichiers sources valident le cache principal. Les tests de sensibilité ont un cache distinct. `--figures-only` permet de modifier le style sans recalculer les valeurs. Les contrôles synthétiques vérifient les probabilités, dénominateurs, fenêtres, décalages non linéaires, niveau du test sur une orbite uniforme et corrections multiples.

Résultat principal : à ±1 s, la poursuite présente la probabilité brute maximale (19,1 %), mais pas d'enrichissement significatif par rapport au niveau local. Les interactions anogénitales (13,2 %) et les approches (11,9 %) montrent un enrichissement qui persiste aux trois échelles locales. Le suivi (15,9 %) passe le test principal, mais reste sensible au choix des blocs de 30 s.
