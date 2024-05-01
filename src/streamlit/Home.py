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
    "Rakuten, fondée en 1997, est devenue une importante plateforme e-commerce avec 1,3 milliards de membres. Le Rakuten Institute of Technology (RIT) propose une problématique de classification de produits basée sur des informations texte et images. Ce défi, commun aux marketplaces, vise à optimiser les recherches de produits en considérant la variabilité des descriptions et des types de produits. La pluralité des classes de produits nécessite une approche de catégorisation manuelle ou basée sur des règles. Ce projet, intéressant sur les plans méthodologique et technique, implique le traitement de données texte et image et l'utilisation de modèles de deep learning, de plus en plus courants dans la résolution de problèmes opérationnels en entreprise."
)
st.write("## Objectifs")
st.write("#### Lors de ce projet, nos objectifs étaient les suivants :")
st.write(
    """- Entrainer des modèles de deep learning sur des données de type texte et image
- Déterminer le meilleur modèle afin de prédire la catégorie d'un produit"""
)
