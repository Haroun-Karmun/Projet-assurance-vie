# Partie dynamique — version simplifiée

Après installation des bibliothèques (voir ci-dessous), lancer partie_dynamique.py dans VS Code.
Le résultat à lire en premier est resultats/synthese.txt.
Dans VS Code, le fichier de travail est partie_dynamique.py.

## Comment lire le fichier

Les imports chargent les outils de calcul.
Six petites fonctions regroupent des calculs utilisés plusieurs fois.
Le travail commence ensuite au commentaire « Q1 — Ajuster Lee–Carter » et se poursuit dans l'ordre jusqu'à Q4.

1. Q1 transforme les décès et les expositions en taux, puis ajuste SVD et Poisson.
2. Q2 analyse les variations de kappa et compare la marche aléatoire à un AR(1).
3. Q3 simule 10 000 futurs et calcule les espérances de vie en 2023.
4. Q4 suit une femme de 60 ans en 2018 et actualise ses paiements possibles.

Les lignes de création des tableaux CSV et des graphiques servent à présenter les résultats.
Pour comprendre la méthode, commencer par les calculs, puis lire les lignes de présentation.

## Les décisions à savoir expliquer

- On utilise les femmes de 20 à 100 ans sur 1990–2018, conformément à l'énoncé.
- D/E mesure les décès par personne-année. La probabilité annuelle est q = 1 − exp(−mu).
- SVD ajuste les log-taux. Poisson ajuste les décès en tenant compte des expositions.
- Les contraintes somme(beta)=1 et somme(kappa)=0 fixent l'échelle et le centrage.
- Après SVD, kappa est réajusté aux décès annuels selon la slide 177. Les résultats intitulés SVD correspondent donc à cette version réajustée, qui ne minimise plus exactement l’erreur logarithmique initiale.
- Les mises à jour Poisson suivent les slides 181–183. Elles partent de la SVD.
- Sur ces données, l'algorithme converge en neuf itérations. Il s'arrête avec un message si la déviance augmente ou si la convergence échoue.
- Poisson donne la plus petite déviance. Pearson/ddl reste proche de 1,51, ce qui invite à discuter la surdispersion et les résidus.
- Les résidus de la marche aléatoire restent autocorrélés au seuil de 5 %. AR(1) sur les variations corrige ce défaut. Ce choix se fonde sur Ljung–Box et les graphiques.
- ADF et Shapiro complètent le diagnostic. Avec 29 années, les conclusions restent fragiles. ADF avec tendance suggère aussi qu'un modèle autour d'une tendance pourrait convenir.
- Les simulations commencent en 2018. La colonne 5 correspond à 2023.
- Pour l'espérance de vie, on utilise les q moyens simulés, comme demandé dans l'énoncé.
- L'espérance complète est l'intégrale de la survie. Elle est calculée exactement dans chaque année sous force constante.
- Après le dernier âge connu, on prolonge par Gompertz, avec pente estimée à 90–100 ans. C'est une hypothèse supplémentaire à mentionner.
- Les taux bruts 2023 sont conservés jusqu'à 109 ans. La classe ouverte 110+ est exclue.
- La rente est à terme échu. Le premier paiement a lieu après un an. Cette convention est à annoncer, car l'énoncé laisse ce point ouvert.
- Chaque scénario suit une cohorte : 60 ans en 2018, 61 ans en 2019, etc.
- Les intervalles et écarts-types couvrent les innovations futures de kappa, avec paramètres estimés fixes. L'incertitude de calibration et le choix du modèle restent hors de ces bandes.

## Ce qui a été simplifié

Le programme principal conserve les deux calibrations, l'analyse des deux kappa, les simulations, les espérances de vie et la comparaison des rentes à XR.
Les fonctions se limitent aux calculs répétés. La régression AR(1) utilise les moindres carrés de statsmodels. Les fichiers intermédiaires de paramètres et de résidus ainsi que les 10 000 prix individuels ne sont plus sauvegardés : les tableaux récapitulatifs et les cinq graphiques suffisent pour le rapport.
L'algorithme Poisson écrit séparément les trois mises à jour du cours.
La recherche de racine SVD utilise directement Newton, avec contrôle du résultat.
Les comparaisons supplémentaires de la première version (FR, rente à terme avancé, troisième modèle temporel, validation glissante, sensibilité aux grands âges) ont été retirées de cette version simplifiée.
La version principale est adaptée aux données fournies. Si elles changent et qu'un contrôle échoue, il faut réexaminer l'ajustement.

La comparaison numérique avec la première version confirme les mêmes paramètres, espérances de vie et prix simulés pour SVD, Poisson et XR.

## Sources

Énoncé Travail2026.pdf, pages 1–2.
Cours LACTU2030 2026–2027, slides 76–77, 110–114 et 168–188.
Données belges HMD fournies dans Deaths_1x1.txt et Exposures_1x1.txt.

La table XR provient de l'annexe de l'arrêté royal du 29 janvier 2013, reproduite par OpenJustice :
https://etaamb.openjustice.be/fr/arrete-royal-du-29-janvier-2013_n2013011073.html
La reproduction est conservée dans data/source_XR.html. Le serveur du Moniteur officiel était inaccessible pendant la préparation.
Vérification effectuée avec data/TablesXRXKbrut.xlsm du dépôt du binôme : les 115 valeurs de survivants, de 0 à 114 ans, sont identiques.
La fermeture du dernier intervalle XR, 113–114 ans, suppose des décès uniformes. Les dernières valeurs ont été vérifiées : l113 = 2 et l114 = 0, ce qui justifie cette fermeture. Les décès et expositions effectivement utilisés dans les données fournies sont strictement positifs ; le script demande de réexaminer les données si ce n’est plus le cas.

## Relancer ailleurs

Dans le terminal de VS Code, depuis ce dossier :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python partie_dynamique.py
```

Sur Mac, après cette installation, `zsh Lancer.command` relance le calcul. Sous Windows, activer l’environnement avec `.venv\Scripts\activate` à la place de la commande `source`.
Les données sont lues depuis le dossier data à côté du script.
La graine 2030 rend les simulations reproductibles.

Le rapport et ses interprétations restent à rédiger. La qualité de la remise dépend aussi de la justification des hypothèses et de la compréhension des résultats.
