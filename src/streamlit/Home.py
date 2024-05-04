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
st.sidebar.progress(0)

# MAIN PAGE

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Rakuten_logo_2.svg/2560px-Rakuten_logo_2.svg.png"
)
st.title(":blue[RAKUTEN FRANCE MULTIMODAL PRODUCT DATA CLASSIFICATION]")

st.write("## Contexte")
st.write(
    "1. > :blue[**Rakuten**], fondée en 1997, est devenue une importante :blue[**plateforme e-commerce**] avec 1,3 milliards de membres. Le Rakuten Institute of Technology (RIT) propose une :blue[**problématique de classification de produits**] basée sur des informations de type :blue[**texte**] et :blue[**image**].\n"
    "2. > Ce défi, commun aux marketplaces, vise à :blue[**optimiser le référencement des produits**] en considérant une variabilité des descriptions textuelles et d'images de produits. Avant l'avènement de méthodes de classification d'images ou de textes, la pluralité des classes de produits nécessitait une fastidieuse approche de :blue[**catégorisation manuelle**] ou basée sur des :blue[**règles grossières**].\n"
    "3. > Ce projet implique donc le :blue[**traitement de données texte et image**], où l'utilisation de :blue[**modèles de deep learning**] est de plus en plus la norme."
)
st.write("## Objectifs")
st.write("##### Lors de ce projet, nos objectifs étaient les suivants :")
st.write(
    "1. > Proposer une :blue[**résolution de la problématique**] grâce à une approche data science driven.\n"
    "2. > Anticiper des :blue[**étapes classiques de preprocessing**] de données image et texte, comprendre leur intérêt et également leurs limites dans un contexte réel.\n"
    "3. > Expérimenter des :blue[**modèles de machine learning et de deep learning**] sur des données de type texte et image et notamment tester les différents apprentissages des modules du cycle Data Scientist de la formation.\n"
    "4. > Déterminer le :blue[**meilleur modèle pour prédire la catégorie d'un produit**] en fonction d'images et textes de produits susceptibles d'être commercialisés sur la plateforme Rakuten.\n"
    "5. > Proposer une :blue[**résolution opérationelle de la problématique**] pour des équipes métier merchandising Rakuten."
)
