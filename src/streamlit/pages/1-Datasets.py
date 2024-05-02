import streamlit as st
import pandas as pd

ROOT = "../../"

# SIDEBAR
pages = [
    "X_train | X_test",
    "y_train",
]
page = st.sidebar.radio("Données :", pages)

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
st.sidebar.progress(1 / 5)

# MAIN PAGE
st.title(":blue[DATASETS]")
st.write("## Introduction")
st.write(
    "Le projet utilise des données provenant d'un défi organisé pour des data scientists, hébergées sur le site challengedataens et fournies par le Rakuten Institute of Technology."
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
    st.write("## Données d'entrée | `X_train ou X_test`")
    st.dataframe(st.session_state.X_train_df[cols].head(), use_container_width=True)

    cols = st.columns(5)
    for index, col in st.session_state.X_train_df.head().iterrows():
        path = f"{ROOT}data/raw/images/image_train/image_{col['imageid']}_product_{col['productid']}.jpg"
        with cols[index]:
            st.image(path, caption=index)
    st.write("Nom d'une image : `image_{imageid}_product_{productid}.jpg`")
else:
    st.write("## Données de sortie | `y_train`")
    st.dataframe(st.session_state.y_train_df.head(), use_container_width=True)
