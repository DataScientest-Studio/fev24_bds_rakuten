import streamlit as st
import pandas as pd

ROOT = "../../"

# SIDEBAR
pages = [
    "Features",
    "Target",
    "Relations entre Datasets",
]
page = st.sidebar.radio("Données :", pages)

# Pour descendre la suite
st.sidebar.markdown(
    """# 
#### """
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
st.sidebar.progress(1 / 5)

# MAIN PAGE
st.title(":blue[DATASETS]")
st.write(
    "La problématique du projet provient d'un data Challenge organisé par l'ENS pour le Rakuten Institute of Technology. Les données sources sont hébergées sur le site challengedataens et regroupent 3 grandes catégories de données : Textes et Images qui seront nos variables features et Classe de produits qui sera notre variable target."
)


if "X_train_df" not in st.session_state:
    st.session_state.X_train_df = pd.read_csv(
        f"{ROOT}data/raw/x_train.csv", index_col=0
    )
if "y_train_df" not in st.session_state:
    st.session_state.y_train_df = pd.read_csv(
        f"{ROOT}data/raw/y_train.csv", index_col=0
    )

if page == pages[0]:
    cols = ["designation", "description", "productid", "imageid"]
    st.write("## Features | `X_train et Images_train`")
    st.dataframe(st.session_state.X_train_df[cols].head(10), use_container_width=True)

    cols = st.columns(10)
    for index, col in st.session_state.X_train_df.head(10).iterrows():
        path = f"{ROOT}data/raw/images/image_train/image_{col['imageid']}_product_{col['productid']}.jpg"
        with cols[index]:
            st.image(path, caption=index)
    st.write(
        "Les features texte sont situées dans un fichier X_train.csv comportant un index, les titres et descriptions des produits ainsi qu'un identifiant image et produit correspondant"
    )
    st.write(
        "Les features image sont situées dans un répertoire image_train qui contient toutes les images avec la dénomination image et produit suivante"
    )
    st.write("Nom d'une image : `image_{imageid}_product_{productid}.jpg`")

elif page == pages[1]:
    st.write("## Target | `y_train`")
    st.dataframe(st.session_state.y_train_df.head(10), use_container_width=True)
    st.write(
        "Les données target sont situées dans un fichier y_train.csv comportant un index et la valeur du code produit"
    )

else:
    st.write("## Relations entre les données")
    relation_image_path = f"{ROOT}reports/Relations_Datasets.jpg"
    image = open(relation_image_path, "rb").read()

    # Afficher l'image
    st.image(image, caption="relations entre les datasets", use_column_width=True)
    st.write(
        "Les deux datasets csv sont liés entre eux grace aux index des deux tables x_train et y_train. Les features images sont liées aux données texte via les valeurs imageid et productid qui sont dans le nom des fichiers et dans les données de X_train. A ce titre X_train fait office de table pivot de tous les datasets."
    )
