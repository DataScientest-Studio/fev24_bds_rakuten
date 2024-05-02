import streamlit as st

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
    st.write("## Classification du problème")
    st.write(
        """Notre projet consiste en une :blue[**classification multimodale de produits**], combinant l'analyse d':blue[**images**] et de :blue[**textes**]. La métrique principale utilisée est le :blue[**"weighted F1 score"**], qui prend en compte la :blue[**précision**] et le :blue[**rappel**], tout en tenant compte des déséquilibres entre les classes. Bien que la :blue[**précision**] soit également utilisée pour certains choix intermédiaires, le "weighted F1 score" offre une évaluation plus :blue[**juste**] et :blue[**représentative**] de la performance des modèles."""
    )
    st.write("## Stratégie")
    st.write(
        """Dans un premier temps, les :blue[**données textuelles**] ont été utilisées pour la modélisation, avec des résultats obtenus rapidement et facilement. Ensuite, des :blue[**modèles sur les données image**] ont été développés, en raison de leur complexité et de leur poids. Cette :blue[**dichotomie entre modèles textes et images**] sera conservée dans la stratégie de modélisation, avec la définition de :blue[**modèles champions**] pour chaque catégorie de données d'entrée. Enfin, des :blue[**modèles combinant les résultats**] des modèles champions seront utilisés pour optimiser les performances (Voting Classifier, stacking, bagging, ...)."""
    )
else:
    st.write("## Quelques modèles")
    type_data = st.selectbox(
        "Choix du type de données d'entrée", ["Texte", "Image", "Texte & image"]
    )
    if type_data == "Texte":
        options_models = ["DNN", "MultinomialNB", "RNN", "XGBoost"]
    elif type_data == "Image":
        options_models = ["Modèle 1", "Modèle 2", "Modèle 3"]
    else:
        options_models = ["Multimodale", "Voting classifier"]
    modele = st.selectbox("Choix du modèle", options_models)
    st.write(modele)
