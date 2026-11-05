# ============================================================
# LACTU2030 - Partie statique : modélisation de la mortalité
# ============================================================
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

pd.set_option("display.precision", 8)
pd.set_option("display.width", 200)


# ---- 1. Chargement et nettoyage des données HMD -------------------------------
from pathlib import Path
DOSSIER = Path(__file__).parent

# Chargement des fichiers texte bruts issus du Human Mortality Database (HMD)
deces = pd.read_csv(DOSSIER / "data" / "Deaths_1x1.txt", sep=r"\s+", skiprows=2, na_values=".")
expo = pd.read_csv(DOSSIER / "data" / "Exposures_1x1.txt", sep=r"\s+", skiprows=2, na_values=".")

# Les âges sont lus au format texte (ex: "110+") : on supprime le "+" et on convertit en nombre
deces["Age"] = deces["Age"].astype(str).str.replace("+", "", regex=False).astype(float)
expo["Age"] = expo["Age"].astype(str).str.replace("+", "", regex=False).astype(float)


# ---- 2. Taux bruts : fonction qui extrait une année et un sexe --
def donnees(annee, sexe):
    d = deces[(deces.Year == annee) & (deces.Age >= 30) & (deces.Age <= 105)]
    e = expo[(expo.Year == annee) & (expo.Age >= 30) & (expo.Age <= 105)]
    tab = pd.DataFrame({
        "age": d.Age.values,
        "mu": d[sexe].values / e[sexe].values,  # taux brut = décès / exposition
        "expo": e[sexe].values,
    })
    # Vérifications de cohérence : exactement 76 âges (30 à 105), âges alignés et taux strictement positifs
    assert len(tab) == 76 and (d.Age.values == e.Age.values).all() and (tab.mu > 0).all()
    return tab

# Chargement des 4 jeux de données d'analyse
f2020 = donnees(2020, "Female")
f2023 = donnees(2023, "Female")
h2020 = donnees(2020, "Male")
h2023 = donnees(2023, "Male")
jeux = {"F2020": f2020, "F2023": f2023, "H2020": h2020, "H2023": h2023}


# ---- 3. Le modèle ----------------------------------------------
def mu_modele(parametres, age):
    a, b, c = parametres
    return np.exp(a + b * age + c * (age - 70) * (age >= 70))


# ---- 4. L'erreur à minimiser (MSE) -----------------------------
def mse(parametres, age, mu):
    return np.mean((mu - mu_modele(parametres, age)) ** 2)
 
 
# ---- 5. Minimisation de la MSE -----------------------------

# Régression linéaire de log(mu) pour initialiser l'optimiseur
def depart(donnees):   
    X = np.column_stack([np.ones(len(donnees)), donnees.age, np.maximum(donnees.age - 70, 0)])
    return np.linalg.lstsq(X, np.log(donnees.mu), rcond=None)[0]

# Ajustement non-linéaire par Nelder-Mead pour minimiser la MSE sur les taux bruts.
def ajuster_mse(donnees):
    res = minimize(mse, depart(donnees), args=(donnees.age.values, donnees.mu.values),
                   method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000})
    assert res.success
    return res.x

# Tableau récapitulatif des paramètres (a, b, c) ajustés par MSE
tableau_mse = pd.DataFrame({nom: ajuster_mse(tab) for nom, tab in jeux.items()},
                           index=["a", "b", "c"]).T

print("\n--- Paramètres optimisés par MSE ---")
print(tableau_mse)
 
# Plot 1 : Graphiques d'ajustement MSE (Taux bruts vs Taux théoriques)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (nom, tab) in zip(axes.flat, jeux.items()):
    ax.plot(tab.age, tab.mu, "ko", ms=3, label="Taux bruts")
    ax.plot(tab.age, mu_modele(tableau_mse.loc[nom], tab.age), "r-", lw=2, label="Taux théoriques")
    ax.set(xlabel="Âge", ylabel=r"$\mu_x$", title=nom)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
plt.tight_layout()
plt.show()


# ---- 6. Espérance de vie (modèle MSE) -----------------------------------
def esperance(parametres, x, age_max=120):
    mu = mu_modele(parametres, np.arange(x, age_max))
    return np.cumprod(np.exp(-mu)).sum()  # somme des probabilités de survie

ages = [40, 50, 60, 70]
esperances_mse = pd.DataFrame(
    {nom: [esperance(tableau_mse.loc[nom], x) for x in ages] for nom in jeux},
    index=[f"e{x}" for x in ages]).T.round(2)


# ---- 7. Ajustement par maximum de log-vraisemblance sous l'hypothèse gaussienne -----------
def negative_log_likelihood(theta, age, mu, expo):
    mu_theo = mu_modele(theta, age)
    sigma_carre = mu_theo / expo  # Var(mu_hat) = mu / E_x (décès ~ Poisson(mu * E_x))
    return 0.5 * np.sum(np.log(2 * np.pi * sigma_carre) + (mu - mu_theo) ** 2 / sigma_carre)

def ajuster_log_like(donnees):
    res = minimize(negative_log_likelihood, depart(donnees),
                   args=(donnees.age.values, donnees.mu.values, donnees.expo.values),
                   method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    assert res.success
    return res.x

# Tableau des paramètres ajustés par Log-Vraisemblance
tableau_log_like = pd.DataFrame({nom: ajuster_log_like(tab) for nom, tab in jeux.items()},
                                index=["a", "b", "c"]).T
print("\n--- Paramètres optimisés par Log-Likelihood ---")
print(tableau_log_like)
print("\n--- Différence entre paramètres (Log-Like - MSE) ---")
print(tableau_log_like - tableau_mse)

# Comparaison des MSE obtenues selon la méthode d'optimisation
comparaison_mse = pd.DataFrame({
    "MSE_directe": [mse(tableau_mse.loc[nom], tab.age, tab.mu) for nom, tab in jeux.items()],
    "MSE_via_likelihood": [mse(tableau_log_like.loc[nom], tab.age, tab.mu) for nom, tab in jeux.items()],
}, index=jeux.keys())

print("\n--- Comparaison des erreurs MSE ---")
print(comparaison_mse)


# ---- 8. Comparaison des espérances de vie et calcul des biais -----------------------------
esperances_log_like = pd.DataFrame(
    {nom: [esperance(tableau_log_like.loc[nom], x) for x in ages] for nom in jeux},
    index=[f"e{x}" for x in ages]).T.round(2)

print("--- Espérances de vie estimées (Modèle MSE) ---")
print(esperances_mse)

print("--- Espérances de vie estimées (Modèle Log-Vraisemblance) ---")
print(esperances_log_like)

print("--- Écart (Log-Vraisemblance - MSE) sur les espérances de vie ---")
print(esperances_log_like - esperances_mse)

def esperance_brute(tab, x, age_max=105):
    mu = tab.mu[(tab.age >= x) & (tab.age <= age_max - 1)]
    return np.cumprod(np.exp(-mu)).sum()

esperances_brutes = pd.DataFrame(
    {nom: [esperance_brute(tab, x) for x in ages] for nom, tab in jeux.items()},
    index=[f"e{x}" for x in ages]).T

print("\n--- Espérances de vie brutes observées ---")
print(esperances_brutes.round(2))

# Calcul des biais de modélisation (Estimé - Observé)
biais_esperance_mse = (esperances_mse - esperances_brutes).round(2)
biais_esperance_log_like = (esperances_log_like - esperances_brutes).round(2)

print("\n--- Biais de l'espérance de vie (MSE) ---")
print(biais_esperance_mse)

print("\n--- Biais de l'espérance de vie (Log-Likelihood) ---")
print(biais_esperance_log_like)


# ---- 9. Graphique comparatif superposé (MSE vs Log-Likelihood) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (nom, tab) in zip(axes.flat, jeux.items()):
    ax.plot(tab.age, tab.mu, "ko", ms=3, label="Taux bruts")
    ax.plot(tab.age, mu_modele(tableau_mse.loc[nom], tab.age), "r-", lw=2, label="Modèle MSE")
    ax.plot(tab.age, mu_modele(tableau_log_like.loc[nom], tab.age), "b--", lw=2,
            label="Modèle Log-Likelihood")
    ax.set(xlabel="Âge", ylabel=r"$\mu_x$", title=nom)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
plt.tight_layout()
plt.show()


# ---- 10. Tarification des rentes viagères -----------------------
# Hypothèses : i = 1.25%, rente de 1000€, durée de 20 ans, âges 50, 60, 70 (femmes 2020 & 2023)
def prix_rente(parametres, x, n=20, i=0.0125, montant=1000, debut=False):
    v = 1 / (1 + i) # facteur d'actualisation
    mu = mu_modele(parametres, np.arange(x, x + n))
    tpx = np.cumprod(np.exp(-mu))  # probabilités de survie temporaires
    if debut:  # rente payée d'avance
        return montant * np.sum(v ** np.arange(0, n) * np.r_[1, tpx[:n - 1]])
    return montant * np.sum(v ** np.arange(1, n + 1) * tpx)  # rente payée à échoir

ages_rentes = [50, 60, 70]
colonnes = [f"x = {x}" for x in ages_rentes]

prix_mse = pd.DataFrame({nom: [prix_rente(tableau_mse.loc[nom], x) for x in ages_rentes]
                         for nom in ["F2020", "F2023"]}, index=colonnes).T
prix_log_like = pd.DataFrame({nom: [prix_rente(tableau_log_like.loc[nom], x) for x in ages_rentes]
                              for nom in ["F2020", "F2023"]}, index=colonnes).T

print("\n--- Prix des rentes (MSE) ---")
print(prix_mse)

print("\n--- Prix des rentes (Log-Likelihood) ---")
print(prix_log_like)



# ---- 11. Comparaison avec la table de mortalité légale ---------
# Table légale belge pour les rentes (FR) : l_x = k s^x g^(c^x)
FR = dict(k=1000048.56, s=0.999669730966, g=0.999951440172, c=1.116792453830)

def lx_FR(x):
    return FR["k"] * FR["s"] ** x * FR["g"] ** (FR["c"] ** x)

print(lx_FR(0))

def prix_rente_legale(x, n=20, i=0.0125, montant=1000, debut=False):
    v = 1 / (1 + i)
    t = np.arange(0, n) if debut else np.arange(1, n + 1)
    tpx = lx_FR(x + t) / lx_FR(x)
    return montant * np.sum(v ** t * tpx)

# Calcul des prix selon la table réglementaire
prix_legal = pd.Series([prix_rente_legale(x) for x in ages_rentes], index=colonnes)


# Calcul des écarts de prix par rapport à la table légale 
ecart_mse = prix_mse - prix_legal  # en euros
ecart_mse_percent = 100 * ecart_mse / prix_legal # en % 
ecart_log_like = prix_log_like - prix_legal
ecart_log_like_percent = 100 * ecart_log_like / prix_legal

# Tableau de synthèse des tarifs
tableau_prix = pd.concat([
    prix_mse.rename(index=lambda n: "MSE_" + n),
    prix_log_like.rename(index=lambda n: "LL_" + n),
    prix_legal.to_frame("Legal_FR").T,
])

print("\n--- Synthèse comparative des prix des rentes (€) ---")
print(tableau_prix.round(0))

print("\n--- Écart relatif au tarif réglementaire FR (%) - Modèle MSE ---")
print(ecart_mse_percent.round(1))

print("\n--- Écart relatif au tarif réglementaire FR (%) - Modèle Log-Likelihood ---")
print(ecart_log_like_percent.round(1))
