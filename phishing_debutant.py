"""
===============================================================================
 MON PREMIER DETECTEUR DE PHISHING - TensorFlow / Keras
-------------------------------------------------------------------------------
 Version DEBUTANT : un petit reseau de neurones qui apprend a reconnaitre
 une URL dangereuse a partir de 10 caracteristiques simples.

 Pour l'executer :
     pip install -r requirements.txt
     python phishing_debutant.py

 Le jeu de donnees est lu dans le fichier urls_phishing_20000.csv
 (colonnes "url" et "label"), place dans le meme dossier que ce script.
===============================================================================
"""

import json
import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Masquer les messages d'information de TensorFlow : ces deux lignes doivent
# etre executees AVANT l'import de TensorFlow, d'ou leur place ici.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # 3 = n'afficher que les erreurs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # resultats identiques a chaque run

import tensorflow as tf      # noqa: E402  (import volontairement apres os.environ)
import keras                 # noqa: E402  (Keras 3, installe avec TensorFlow)
from keras import layers     # noqa: E402

from outils_url import mesurer_url   # noqa: E402  (les 10 variables)

# On fixe le "hasard" pour obtenir les memes resultats a chaque execution.
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# ETAPE 1 - CHARGER LE JEU DE DONNEES (fichier CSV)
# ----------------------------------------------------------------------------
# Le fichier doit contenir deux colonnes : "url" et "label"
# (0 = legitime, 1 = phishing). Il doit etre dans le meme dossier que ce
# script, ou bien indiquez son chemin complet dans FICHIER_CSV.
# ============================================================================
FICHIER_CSV = "urls_phishing_20000.csv"

if not os.path.exists(FICHIER_CSV):
    raise SystemExit(f"Fichier introuvable : {FICHIER_CSV}. Placez-le a cote "
                     "de ce script ou corrigez FICHIER_CSV.")

df = pd.read_csv(FICHIER_CSV, usecols=["url", "label"])

# Nettoyage : lignes vides, espaces, etiquettes autres que 0 / 1
df = df.dropna()
df["url"] = df["url"].astype(str).str.strip()
df = df[df["url"] != ""]
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])].astype({"label": int})

# On retire les doublons : sinon une meme URL peut se retrouver a la fois
# dans l'apprentissage et dans le test. Une URL presente avec les deux
# etiquettes est ambigue : on l'ecarte.
ambigues = df.groupby("url")["label"].nunique()
df = df[~df["url"].isin(ambigues[ambigues > 1].index)]
df = df.drop_duplicates(subset="url").reset_index(drop=True)

print("Fichier lu :", FICHIER_CSV)
print("Nombre d'URL :", len(df))
print(df["label"].value_counts().rename({0: "legitimes", 1: "phishing"}))
print(df.head(3).to_string(index=False))


# ============================================================================
# ETAPE 2 - TRANSFORMER CHAQUE URL EN 10 NOMBRES (les variables explicatives)
# ----------------------------------------------------------------------------
# Un reseau de neurones ne comprend pas le texte : il faut lui donner des
# nombres. On mesure donc 10 choses simples sur chaque URL.
# ============================================================================
# Les fonctions extraire_hote, est_adresse_ip et mesurer_url sont dans
# outils_url.py, pour etre reutilisees par l'application Streamlit.


X = pd.DataFrame([mesurer_url(u) for u in df["url"]])   # variables explicatives
y = df["label"].to_numpy(dtype="float32")               # variable cible

print("\nExemple de mesures pour la 1re URL :")
print(X.head(1).to_string(index=False))


# ============================================================================
# ETAPE 3 - SEPARER APPRENTISSAGE ET TEST
# ----------------------------------------------------------------------------
# Regle d'or : on ne teste JAMAIS sur les donnees qui ont servi a apprendre.
# 80 % pour apprendre, 20 % pour verifier honnetement.
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(dtype="float32"), y,
    test_size=0.2, random_state=42, stratify=y)

print(f"\nApprentissage : {len(X_train)} URL   |   Test : {len(X_test)} URL")


# ============================================================================
# ETAPE 4 - METTRE LES NOMBRES A LA MEME ECHELLE
# ----------------------------------------------------------------------------
# "longueur" vaut ~80 et "https" vaut 0 ou 1 : sans normalisation, le reseau
# donnerait trop d'importance aux grands nombres.
# ATTENTION : on calcule la moyenne sur le TRAIN seulement, jamais sur le test.
# ============================================================================
normaliseur = StandardScaler()
X_train = normaliseur.fit_transform(X_train)
X_test = normaliseur.transform(X_test)


# ============================================================================
# ETAPE 5 - CONSTRUIRE LE RESEAU DE NEURONES
# ----------------------------------------------------------------------------
# Trois couches empilees :
#   - Dense(16) : 16 neurones qui cherchent des combinaisons de nos 10 mesures
#   - Dense(8)  : 8 neurones qui resument encore
#   - Dense(1, sigmoid) : 1 seul neurone qui sort une probabilite entre 0 et 1
# ============================================================================
modele = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),          # eteint 20 % des neurones : evite le "par coeur"
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

modele.compile(
    optimizer="adam",                 # l'algorithme qui ajuste les poids
    loss="binary_crossentropy",       # la "note d'erreur" pour un choix 0/1
    metrics=["accuracy"],             # ce qu'on affiche pour se rassurer
)

modele.summary()


# ============================================================================
# ETAPE 6 - ENTRAINER LE MODELE
# ----------------------------------------------------------------------------
# Une "epoch" = le modele a vu une fois toutes les donnees d'apprentissage.
# EarlyStopping arrete tout seul quand le modele ne progresse plus.
# ============================================================================
arret_auto = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)

historique = modele.fit(
    X_train, y_train,
    validation_split=0.2,   # 20 % du train sert a surveiller l'apprentissage
    epochs=50,
    batch_size=32,
    callbacks=[arret_auto],
    verbose=2,
)


# ============================================================================
# ETAPE 7 - EVALUER : est-ce que ca marche vraiment ?
# ============================================================================
perte, exactitude = modele.evaluate(X_test, y_test, verbose=0)
print(f"\nExactitude sur le jeu de test : {exactitude:.1%}")

probas = modele.predict(X_test, verbose=0).ravel()
predictions = (probas >= 0.5).astype(int)

print("\nMatrice de confusion :")
print(pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index=["Vrai: legitime", "Vrai: phishing"],
    columns=["Predit: legitime", "Predit: phishing"]))

print("\n", classification_report(y_test, predictions,
                                  target_names=["legitime", "phishing"],
                                  digits=3))


# ============================================================================
# ETAPE 8 - UTILISER LE MODELE SUR UNE NOUVELLE URL
# ============================================================================
modele.save("mon_modele_phishing.keras")
# CORRECTION : sans le normaliseur, le modele sauvegarde est inutilisable,
# car chaque nouvelle URL doit etre mise a la meme echelle que le train.
joblib.dump(normaliseur, "mon_normaliseur.joblib")

# ETAPE 9 - Enregistrer les resultats pour l'application Streamlit
with open("metriques.json", "w", encoding="utf-8") as f:
    json.dump({
        "exactitude": float(exactitude),
        "y_test": y_test.astype(int).tolist(),
        "probas": probas.round(5).tolist(),
        "historique": {k: [float(v) for v in vals]
                       for k, vals in historique.history.items()},
        "nb_apprentissage": int(len(X_train)),
        "nb_test": int(len(X_test)),
    }, f)
print("Modele, normaliseur et metriques.json enregistres.")


def analyser(url):
    mesures = pd.DataFrame([mesurer_url(url)])
    mesures = normaliseur.transform(mesures.to_numpy(dtype="float32"))
    proba = float(modele.predict(mesures, verbose=0)[0][0])
    verdict = "DANGEREUX" if proba >= 0.5 else "PROBABLEMENT SUR"
    return f"{proba:6.1%}  {verdict}  <-  {url}"


print("\n--- Test sur de nouvelles URL ---")
urls_connues = set(df["url"])
for essai in [
    "https://www.amazon.com/compte/aide-77",
    "http://paypal-verify.secure-paypal.xyz/login/session4821?cmd=verify&id=91",
    "https://www.impots.com/contact",
    "http://192.168.12.44/update/session7712?cmd=update&id=7712",
]:
    # CORRECTION : on signale si l'URL "nouvelle" figurait deja dans les donnees
    deja = "  (deja vue a l'entrainement)" if essai in urls_connues else ""
    print(analyser(essai) + deja)