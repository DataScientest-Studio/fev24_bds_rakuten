from pyexpat import model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "../../"

# SIDEBAR
pages = [
    "la stratégie",
    "les modèles",
]
page = st.sidebar.radio("Voir :", pages)
# Pour descendre la suite
st.sidebar.markdown(
    """# 
##### 
# """
)
st.sidebar.info(
    """:grey[**PROJET DATASCIENTEST :**]

:grey[Rakuten France multimodal product data classification]

:grey[**PROMOTION :**]

:grey[fev24_bootcamp_ds]

:grey[**PARTICIPANTS :**]

- :grey[Komla CHOKKI]
- :grey[Jérémy RAVA]
- :grey[Bruno VALETTE]
"""
)
st.sidebar.progress(3 / 5)

# MAIN PAGE
st.title(":blue[MODÉLISATION]")

if page == pages[0]:
    st.write("### Classification du problème et Métrique de performance")
    st.write(
        " 1. > Notre projet consiste en une :blue[**classification multimodale de produits**], appelant à une modélisation qui pourrait combiner l'analyse d':blue[**images**] et de :blue[**textes**].\n"
        " 2. > En raison de la distribution déséquilibrée de la donnée Target, la métrique principale utilisée est le :blue[**weighted F1 score**], qui prend en compte la :blue[**précision**] et le :blue[**rappel**].\n"
        " 3. > La :blue[**précision**] ayant été tout au long de notre projet utilisée pour certains choix intermédiaires de modèles, le :blue[**weighted F1 score**] offre une évaluation plus :blue[**juste**] et :blue[**représentative**] de la performance des modèles, notamment sur les échantillons de validation.\n"
    )
    st.write("### Stratégie")
    st.write(
        "1. > Dans un premier temps, seules les :blue[**données textuelles**] ont été utilisées pour la modélisation. Le developpement des modèles a été plus simple et a demandé moins de ressources machines ou temporelles pour obtenir des résultats. Nous nous sommes principalement appuyés sur les recommandations du site Scikit-Learn afin de tester les modèles potentiellement les plus performants.\n"
        "2. > Par la suite, des :blue[**modèles sur les données image**] ont été développés, en raison de leur complexité et des ressources qu'ils nécessitaient.\n"
        "3. > En raison de la complexité et de la différence de nature des données, cette :blue[**dichotomie entre modèles textes et images**] sera conservée dans la stratégie de modélisation, avec la définition de :blue[**modèles champions**] pour chaque catégorie de données d'entrée. Enfin, nous utiliserons des :blue[**modèles combinant les résultats**] des meilleurs modèles texte et image pour optimiser les performances (Voting Classifier, stacking, bagging, ...)."
    )
    st.write("### Modèles Expérimentés")
    st.write("#### Modèles Texte")
    st.write(
        "1. > Après :blue[**pre-processing**] des données textes (nettoyage ponctuations, balises HTML, traduction, regex, stopwords, lemmatisation), nous avons expérimenté l'ensemble de nos modèles sur les données :blue[**vectorisées et rééquilbrées**] (oversampling SMOTE sur la majorité des modèles).\n"
        "2. > Nous avons d'abord expérimenté des modèles de :blue[**Machine Learning**] : KNN, Random Forest, XGBoost, Modèles Naive Bayes qui ont montré des performances intéressantes. Nous avons également expérimenté des modèles de :blue[**Deep Learning**] qui ont globalement montré de meilleures performances que ceux de ML.\n"
    )
    st.write("#### Modèles Image")
    st.write(
        "1. > De la même façon, nous avons opéré à un :blue[**pre-processing**] des données images (cropping, resizing) avant de modéliser. Nous avons utilisé plusieurs transformations des données image pour modélisation : considération des pixels et :blue[**ACP**], utilisation des caratéristiques :blue[**HOG**], ou techniques de :blue[**convolutions pour les réseaux de Neurones**]. Le reéchantillonage a été fait de manières diverse (undersampling données pixels, oversampling données HOG, datageneration pour NN).\n"
        "2. > De nombreux modèles de :blue[**Machine Learning**] (Logistique, RF, XGBoost, SVM) ont été expérimenté sur les données pixels réduites (ACP 90% de la variance). Nous avons également pratiqué ce type de modèles (RF, SVC) sur les caractéristiques HOG des images. (avec optimisation des hyper paramètres - Gridsearch).\n"
        "3. > Nous avons enfin expérimenté des modèles :blue[**CNN**] et, après essais et recherches, décidé d'utiliser des :blue[**modèles pré entrainés**] sur ces problématiques (ResNet, EfficentNet).\n"
    )

    st.image(
        f"{ROOT}reports/all_modeles_results.png",
        use_column_width=True,
        caption="Grille des principaux modèles expérimentés et leurs performances",
    )
    st.write(
        "1. > Avec la sélection de modèles champions image et texte, nous avons enfin expérimenté des méthodes de :blue[**combinaison de modèles**] (Voting, Stacking).\n"
        "2. > L'idée de notre stratégie de modélisation étant de choisir :blue[**1 modèle texte**] et :blue[**1 modèle image**] en entrée de la technique d'ensemble (parmi les :blue[**2 types de modèles Machine Learning ou Deep Learning**]).\n"
        "3. > In fine, nous avons expérimenté une approche plus simpliste d'utiliser des :blue[**features**] issues des images et des textes pour entrainer un :blue[**seul modèle de Machine Learning**] : RF.\n"
    )
else:
    st.write("## Quelques modèles")
    type_data = st.selectbox(
        "Choix du type de données d'entrée :", ["Texte", "Image", "Texte & image"]
    )
    if type_data == "Texte":
        options_models = ["DNN", "MultinomialNB", "RNN", "XGBoost"]
    elif type_data == "Image":
        options_models = ["CNN_EfficientNetB0", "RF_HOG", "Modèle 3"]
    else:
        options_models = ["Multimodale", "Voting classifier"]
    modele = st.selectbox("Choix du modèle :", options_models)

    if modele in ["MultinomialNB", "XGBoost", "RF_HOG"]:
        df = pd.read_csv(f"{ROOT}reports/modeles/cv_results_{modele}.csv")
        fig, ax = plt.subplots()
        ax.plot(
            df["rank_test_score"].sort_values(ascending=False), df["mean_test_score"]
        )
        ax.set_xlabel("Run")
        ax.set_ylabel("Mean F1 score")
        st.write(fig)
        st.image(
            f"{ROOT}reports/modeles/training_confusion_matrix_{modele}.png",
            use_column_width=True,
            caption=f"Confusion Matrix of {modele}",
        )
    elif modele in ["DNN", "RNN", "CNN_EfficientNetB0"]:
        df_val_acc = pd.read_csv(
            f"{ROOT}reports/modeles/validation_accuracy_{modele}.csv"
        )
        df_val_f1 = pd.read_csv(
            f"{ROOT}reports/modeles/validation_f1_score_{modele}.csv"
        )
        df_val_loss = pd.read_csv(f"{ROOT}reports/modeles/validation_loss_{modele}.csv")
        df = df_val_acc.copy()
        df["val_f1_score"] = df_val_f1["val_f1_score"]
        df["val_loss"] = df_val_loss["val_loss"]
        fig, axs = plt.subplots(1, 3, figsize=(15, 7))
        axs[0].plot(df.index, df["val_accuracy"])
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Validation accuracy")

        axs[1].plot(df.index, df["val_f1_score"])
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Validation F1 score")

        axs[2].plot(df.index, df["val_loss"])
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Validation loss")
        st.write(fig)
