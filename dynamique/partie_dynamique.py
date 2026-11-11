"""LACTU2030 — Partie dynamique.
Femmes belges, 20–100 ans, 1990–2018.
Exécuter le fichier entier. Les résultats sont dans resultats/.
"""
from pathlib import Path
import os

DOSSIER = Path(__file__).resolve().parent
RESULTATS = DOSSIER / "resultats"
RESULTATS.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RESULTATS / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.special import xlogy
from scipy.stats import shapiro
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

ages = np.arange(20, 101)
annees = np.arange(1990, 2019)
nombre_simulations = 10000
taux_interet = 0.0125


# Fonctions utilisées plusieurs fois
def lire_matrice(nom, ages_voulus, annees_voulues):
    # Dans les deux fichiers fournis, l'en-tête suit deux lignes de texte.
    table = pd.read_csv(DOSSIER / "data" / nom, sep=r"\s+",
                        skiprows=2, dtype={"Age": str}, na_values=["."])
    table = table[table["Age"].isin(ages_voulus.astype(str))].copy()
    table["Age"] = table["Age"].astype(int)
    matrice = table.pivot(index="Age", columns="Year", values="Female")
    matrice = matrice.reindex(index=ages_voulus, columns=annees_voulues)
    valeurs = matrice.to_numpy(float)
    if not np.isfinite(valeurs).all() or np.any(valeurs <= 0):
        raise ValueError("Vérifier les données : valeur absente ou non positive.")
    return valeurs


def normaliser(alpha, beta, kappa):
    # Ces transformations conservent alpha + beta*kappa.
    moyenne = kappa.mean()
    somme = beta.sum()
    if abs(somme) < 1e-10:
        raise ValueError("Normalisation impossible : somme des beta nulle.")
    alpha = alpha + beta * moyenne
    kappa = (kappa - moyenne) * somme
    beta = beta / somme
    return alpha, beta, kappa


def deviance(deces, attendus):
    return 2 * np.sum(xlogy(deces, deces / attendus) - deces + attendus)


def sauvegarder_figure(nom):
    plt.tight_layout()
    plt.savefig(RESULTATS / nom, dpi=150)
    plt.close()


def ajuster_dynamique(kappa, modele):
    variations = np.diff(kappa)
    if modele == "Marche":
        constante = variations.mean()
        phi = 0.0
        residus = variations - constante
        sigma = residus.std(ddof=1)  # Dénominateur n-2, slide 188.
    else:
        # Régression : variation suivante = constante + phi*variation précédente.
        regression = OLS(variations[1:], add_constant(variations[:-1])).fit()
        constante, phi = regression.params
        residus = regression.resid
        sigma = np.sqrt(regression.mse_resid)
        if abs(phi) >= 1:
            raise ValueError("Les variations AR(1) ne sont pas stationnaires.")
    return constante, phi, sigma, residus


def esperance_vie(mu, age):
    # Prolongement Gompertz après le dernier âge disponible.
    ages_connus = np.arange(20, 20 + len(mu))
    grands_ages = (ages_connus >= 90) & (ages_connus <= 100)
    pente = np.polyfit(ages_connus[grands_ages], np.log(mu[grands_ages]), 1)[0]
    if pente <= 0:
        raise ValueError("Vérifier la pente de mortalité aux grands âges.")
    ages_futurs = np.arange(ages_connus[-1] + 1, 141)
    suite = mu[-1] * np.exp(pente * (ages_futurs - ages_connus[-1]))
    mu = np.concatenate([mu, suite])[age - 20:]
    survie_fin = np.exp(-np.cumsum(mu))
    survie_debut = np.r_[1.0, survie_fin[:-1]]
    if survie_fin[-1] > 1e-10:
        raise ValueError("La survie à 141 ans doit être négligeable.")
    # Temps moyen vécu dans chaque année, avec une force constante.
    return np.sum(survie_debut * (-np.expm1(-mu)) / mu)


# Q1 — Ajuster Lee–Carter
deces = lire_matrice("Deaths_1x1.txt", ages, annees)
expositions = lire_matrice("Exposures_1x1.txt", ages, annees)
mu_observe = deces / expositions

# SVD : retirer le niveau moyen par âge, puis garder le premier facteur.
log_mu = np.log(mu_observe)
alpha = log_mu.mean(axis=1)
matrice_centree = log_mu - alpha[:, None]
u, valeurs, vt = np.linalg.svd(matrice_centree, full_matrices=False)
alpha, beta, kappa = normaliser(alpha, u[:, 0], valeurs[0] * vt[0])

# Slide 177 : ajuster kappa pour reproduire les décès annuels observés.
for t in range(len(annees)):
    def ecart(k):
        return np.sum(expositions[:, t] * np.exp(alpha + beta * k)) - deces[:, t].sum()

    def derivee(k):
        return np.sum(expositions[:, t] * np.exp(alpha + beta * k) * beta)

    kappa[t] = newton(ecart, kappa[t], fprime=derivee)
    if abs(ecart(kappa[t])) > 1e-5:
        raise RuntimeError("L'ajustement des décès annuels a échoué.")

alpha, beta, kappa = normaliser(alpha, beta, kappa)
modeles = {"SVD": (alpha.copy(), beta.copy(), kappa.copy())}

# Poisson : mises à jour successives des paramètres (slides 181–183).
for iteration in range(1000):
    log_mu_avant = alpha[:, None] + beta[:, None] * kappa
    attendus = expositions * np.exp(log_mu_avant)
    deviance_avant = deviance(deces, attendus)

    alpha += (deces - attendus).sum(axis=1) / attendus.sum(axis=1)
    attendus = expositions * np.exp(alpha[:, None] + beta[:, None] * kappa)

    numerateur = ((deces - attendus) * beta[:, None]).sum(axis=0)
    denominateur = (attendus * beta[:, None]**2).sum(axis=0)
    kappa += numerateur / denominateur
    attendus = expositions * np.exp(alpha[:, None] + beta[:, None] * kappa)

    numerateur = ((deces - attendus) * kappa).sum(axis=1)
    denominateur = (attendus * kappa**2).sum(axis=1)
    beta += numerateur / denominateur

    alpha, beta, kappa = normaliser(alpha, beta, kappa)
    log_mu_apres = alpha[:, None] + beta[:, None] * kappa
    nouvelle_deviance = deviance(deces, expositions * np.exp(log_mu_apres))
    if not np.isfinite(nouvelle_deviance) or nouvelle_deviance > deviance_avant + 1e-7:
        raise RuntimeError("L'algorithme Poisson doit être stabilisé pour ces données.")
    if np.max(np.abs(log_mu_apres - log_mu_avant)) < 1e-9:
        break
else:
    raise RuntimeError("L'algorithme Poisson n'a pas convergé.")
modeles["Poisson"] = (alpha.copy(), beta.copy(), kappa.copy())

comparaison = []
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for nom, (alpha, beta, kappa) in modeles.items():
    mu = np.exp(alpha[:, None] + beta[:, None] * kappa)
    attendus = expositions * mu
    residus = (deces - attendus) / np.sqrt(attendus)
    degres_liberte = deces.size - (2 * len(ages) + len(annees) - 2)
    comparaison.append({"methode": nom, "deviance": deviance(deces, attendus),
        "RMSE_log": np.sqrt(np.mean((np.log(mu_observe) - np.log(mu))**2)),
        "Pearson_sur_ddl": np.sum(residus**2) / degres_liberte})
    axes[0].plot(ages, alpha, label=nom)
    axes[1].plot(ages, beta, label=nom)
    axes[2].plot(annees, kappa, label=nom)
for ax, titre in zip(axes, ["Profil α par âge", "Sensibilité β par âge", "Indice κ par année"]):
    ax.set_title(titre)
    ax.legend()
sauvegarder_figure("q1_parametres.png")
comparaison = pd.DataFrame(comparaison)
comparaison.to_csv(RESULTATS / "q1_comparaison.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, nom in zip(axes, modeles):
    alpha, beta, kappa = modeles[nom]
    attendus = expositions * np.exp(alpha[:, None] + beta[:, None] * kappa)
    residus = (deces - attendus) / np.sqrt(attendus)
    image = ax.imshow(residus, origin="lower", aspect="auto", cmap="RdBu_r",
                      extent=[1989.5, 2018.5, 19.5, 100.5], vmin=-5, vmax=5)
    ax.set(title=nom, xlabel="Année", ylabel="Âge")
fig.colorbar(image, ax=axes.tolist(), label="Résidu de Pearson (échelle ±5)")
fig.savefig(RESULTATS / "q1_residus.png", dpi=150, bbox_inches="tight")
plt.close()


# Q2 — Analyser kappa et choisir sa dynamique
diagnostics = []
dynamiques = {}
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ligne, (nom, (_, _, kappa)) in enumerate(modeles.items()):
    variations = np.diff(kappa)
    axes[ligne, 0].plot(annees[1:], variations, marker="o")
    axes[ligne, 0].set_title(f"Variations de κ — {nom}")
    plot_acf(variations, lags=8, ax=axes[ligne, 1], title=f"Autocorrélation — {nom}")

    for modele in ["Marche", "AR1"]:
        constante, phi, sigma, residus = ajuster_dynamique(kappa, modele)
        retard_modele = int(modele == "AR1")
        p_lb = acorr_ljungbox(residus, lags=[5], model_df=retard_modele).lb_pvalue.iloc[0]
        diagnostics.append({"methode": nom, "modele": modele,
            "constante": constante, "phi": phi, "sigma": sigma,
            "Ljung_Box_p": p_lb, "Shapiro_p": shapiro(residus).pvalue,
            "ADF_niveau_p": adfuller(kappa, regression="ct", maxlag=3, autolag="AIC")[1],
            "ADF_differences_p": adfuller(variations, maxlag=3, autolag="AIC")[1]})

    # Sur ces données, la marche laisse de l'autocorrélation (p < 5 %).
    # AR(1) permet à la variation suivante de dépendre de la précédente.
    marche, ar1 = diagnostics[-2:]
    choix = "AR1" if marche["Ljung_Box_p"] < 0.05 else "Marche"
    if choix == "AR1" and ar1["Ljung_Box_p"] < 0.05:
        raise RuntimeError("Revoir le modèle temporel : résidus autocorrélés.")
    dynamiques[nom] = ajuster_dynamique(kappa, choix)[:3]
sauvegarder_figure("q2_diagnostics.png")
pd.DataFrame(diagnostics).to_csv(RESULTATS / "q2_diagnostics.csv", index=False)


# Q3 — Simuler les mortalités en 2023 et les espérances de vie
xr = pd.read_csv(DOSSIER / "data" / "table_XR.csv").set_index("age")["lx"]
ages_reels = np.arange(20, 110)
mu_reel = lire_matrice("Deaths_1x1.txt", ages_reels, [2023]).ravel()
mu_reel /= lire_matrice("Exposures_1x1.txt", ages_reels, [2023]).ravel()
trajectoires = {}
esperances = []
fig, ax = plt.subplots(figsize=(8, 5))

for nom, (alpha, beta, kappa) in modeles.items():
    constante, phi, sigma = dynamiques[nom]
    rng = np.random.default_rng(2030)
    chemins = np.zeros((nombre_simulations, 21))
    chemins[:, 0] = kappa[-1]
    variation = np.full(nombre_simulations, kappa[-1] - kappa[-2])
    for t in range(1, 21):
        choc = rng.normal(0, sigma, nombre_simulations)
        variation = constante + phi * variation + choc
        chemins[:, t] = chemins[:, t - 1] + variation
    trajectoires[nom] = chemins

    # La colonne 0 est 2018, la colonne 5 est donc 2023.
    mu_2023 = np.exp(alpha + chemins[:, 5, None] * beta)
    q_2023 = 1 - np.exp(-mu_2023)
    statistiques = pd.DataFrame({"age": ages})
    for variable, valeurs_simulees in [("mu", mu_2023), ("q", q_2023)]:
        statistiques[variable + "_moyenne"] = valeurs_simulees.mean(axis=0)
        statistiques[variable + "_ecart_type"] = valeurs_simulees.std(axis=0, ddof=1)
        statistiques[variable + "_p025"] = np.quantile(valeurs_simulees, .025, axis=0)
        statistiques[variable + "_p975"] = np.quantile(valeurs_simulees, .975, axis=0)
    statistiques.to_csv(RESULTATS / f"q3_mortalite_{nom}.csv", index=False)
    ax.plot(ages, statistiques.q_moyenne, label=nom)
    ax.fill_between(ages, statistiques.q_p025, statistiques.q_p975, alpha=.2)
    mu_moyen_equivalent = -np.log(1 - q_2023.mean(axis=0))
    for age in [20, 30, 40, 50, 60, 70]:
        esperances.append({"table": nom, "age": age,
                          "esperance": esperance_vie(mu_moyen_equivalent, age)})

for age in [20, 30, 40, 50, 60, 70]:
    esperances.append({"table": "Brut 2023", "age": age,
                      "esperance": esperance_vie(mu_reel, age)})
    survivants = xr.loc[age:114].to_numpy(float)
    survie = survivants / survivants[0]
    forces = -np.log(survivants[1:-1] / survivants[:-2])
    vie_xr = np.sum(survie[:-2] * (1 - np.exp(-forces)) / forces)
    vie_xr += survie[-2] / 2  # Fermeture uniforme de la dernière année XR.
    esperances.append({"table": "XR", "age": age, "esperance": vie_xr})

esperances = pd.DataFrame(esperances).pivot(index="age", columns="table", values="esperance")
esperances.to_csv(RESULTATS / "q3_esperances_vie.csv")
ax.plot(ages_reels, 1 - np.exp(-mu_reel), color="black", label="Brut 2023")
ax.set(xlabel="Âge", ylabel="Probabilité annuelle q", yscale="log",
       title="Mortalité 2023 et intervalles ponctuels à 95 %")
ax.legend()
sauvegarder_figure("q3_mortalite.png")


# Q4 — Rente de 1000 euros, 20 ans, à terme échu
resultats_rente = []
fig, ax = plt.subplots(figsize=(8, 5))
for nom, (alpha, beta, _) in modeles.items():
    survie = np.ones(nombre_simulations)
    prix = np.zeros(nombre_simulations)
    for t in range(20):
        indice_age = 60 + t - 20
        kappa = trajectoires[nom][:, t]
        mu = np.exp(alpha[indice_age] + beta[indice_age] * kappa)
        survie *= np.exp(-mu)
        prix += 1000 * survie / (1 + taux_interet)**(t + 1)
    resultats_rente.append({"table": nom, "moyenne": prix.mean(),
        "ecart_type": prix.std(ddof=1), "p01": np.quantile(prix, .01),
        "p05": np.quantile(prix, .05), "p95": np.quantile(prix, .95),
        "p99": np.quantile(prix, .99)})
    ax.hist(prix, bins=40, alpha=.5, density=True, label=nom)

prix_xr = 0.0
for t in range(1, 21):
    survie = xr.loc[60 + t] / xr.loc[60]
    prix_xr += 1000 * survie / (1 + taux_interet)**t
resultats_rente.append({"table": "XR", "moyenne": prix_xr, "ecart_type": 0.0})
rentes = pd.DataFrame(resultats_rente)
rentes.to_csv(RESULTATS / "q4_rentes.csv", index=False)
ax.axvline(prix_xr, color="black", linestyle="--", label="XR")
ax.set(xlabel="Valeur actuarielle (€)", ylabel="Densité", title="Rente à terme échu")
ax.legend()
sauvegarder_figure("q4_rentes.png")

resume = "\n\n".join([
    "Q1 — Qualité des ajustements", comparaison.round(4).to_string(index=False),
    "Q2 — Paramètres (constante, phi, sigma)", str(dynamiques),
    "Q3 — Espérances de vie de période", esperances.round(3).to_string(),
    "Q4 — Rentes à terme échu", rentes.round(3).to_string(index=False),
    "Hypothèses : paramètres fixés, fermeture Gompertz après le dernier âge.",
    "Les intervalles reflètent les innovations futures de kappa."
])
(RESULTATS / "synthese.txt").write_text(resume, encoding="utf-8")
print(resume)
print("\nRésultats enregistrés dans", RESULTATS)
