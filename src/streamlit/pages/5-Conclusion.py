import streamlit as st

# SIDEBAR
# Pour descendre la suite
st.sidebar.markdown(
    """# 
# 
# 
# 
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
st.sidebar.progress(5 / 5)

# MAIN PAGE
st.title(":blue[CONCLUSION]")

st.write("--------------------")
st.write("### Apprentissage")
st.write(
    "Le projet nous a permi d'entrevoir via un cas concret un pan important des :blue[**méthodes et techniques abordées lors de la formation**]. Métriques de performance adaptées, pre-processing de données texte et image, méthode de re-échantillonage pour prévenir le sur-apprentissage, construction d'outils de machine Learning, optimisation d'hyper paramètres, construction d'outils de deep learning : réseaux de neurones et transfert learning.\n"
)
st.write("--------------------")
st.write("### Améliorations")
st.write(
    "Nous avons tout au long du projet pu entrevoir des améliorations que nous n'avons pas pu expérimenter par limite de temps, de ressources ou d'expérience mais que nous voulions partager pour auditer leur pertinence:\n"
    "1. > :blue[**Entrainement plus long**] des modèles de deep learning pré-entrainés. (Recommandations Resnet Max 150 epochs). Nos limites machines nous ont permi d'entrainer seulement sur 10 Epochs.\n"
    "2. > Construction de :blue[**modèles spécifiques**] pour les :blue[**classes les moins bien prédites**] (ie : 2583 - Piscines & Accessoires F1-Score=0.78) et integration de ces modèles au Voting / Stacking Classifier.\n"
    "3. > Expérimenter d':blue[**autres modèles image**] qui font autorité sur le sujet.  [ImageNet Benchmark](https://paperswithcode.com/sota/image-classification-on-imagenet)\n"
    "4. > :blue[**Reinforcement Learning**] sur validations / corrections de notre modèle par des équipes Métier e-Merchandising Rakuten."
)
st.write("--------------------")
st.write("### Operationalité")
st.write(
    "1. > Globalement, notre modèle final nous semble avoir une :blue[**performance acceptable**] pour des équipes métier. :blue[**6% d'erreurs**] peut être de manière réaliste géré opérationnellement sans pénaliser fortement les ressources humaines.\n"
    "2. > Notre modèle a de plus l'avantage de fournir un estimateur de la qualité de classification. La :blue[**mise en place d'un seuil**] en deçà duquel les ressources métier peuvent :blue[**faire du controle/correction**] semble intéressant. Si cette correction est de plus interprétable par le modèle pour modifier et ajuster ses paramètres, cela permettrait de rendre les équipes métier acteurs du projet data et de leur faire anticiper la :blue[**complémentarité Homme x Machine**] dans ce types de projet.\n"
    "3. > Notre modèle s':blue[**appuie sur un existant**] mais la particularité des marketplaces est de connaitre des :blue[**évolutions de classification produits**] très sensibles au temps / aux modes. Ainsi, un contrôle d'experts humains est nécessaire pour anticiper la désuétude ou la nouveauté des classes sur la plateforme. \n "
)
