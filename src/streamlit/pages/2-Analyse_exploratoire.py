import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

ROOT = "../../"


def create_word_cloud(preprocessing, target, code):
    col_target = "text"
    if preprocessing == "0 - Sans":
        df = st.session_state.X_train_df.copy()
        # Join texts
        df["text"] = np.where(
            df["description"].isna(),
            df["designation"].astype(str),
            df["designation"].astype(str) + " " + df["description"].astype(str),
        )
        df["prdtypecode"] = target
        stopwords = []
    elif preprocessing == "1 - Ponctuation, décodage HTML":
        df = st.session_state.X_train_preprocessed_df.copy()
        stopwords = []
    elif preprocessing == "2 - Traduction, regex charactères spéciaux":
        df = st.session_state.X_train_preprocessed_df.copy()
        stopwords = st.session_state.stop_words
    else:
        col_target = "lemmes"
        df = st.session_state.X_train_preprocessed_df.copy()
        stopwords = st.session_state.stop_words

    # Process each unique prdtypecode
    code_df = df[df["prdtypecode"] == code]

    # Remove words from the 'text' column
    code_df[col_target] = code_df[col_target].apply(
        lambda x: " ".join(
            [word for word in x.split() if word.lower() not in stopwords]
        )
    )

    total_text = " ".join(code_df[col_target])
    word_counts = Counter(total_text.split())

    # Sort the word counts in descending order
    word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # PLOTS
    # WORDCLOUD
    fig2, ax2 = plt.subplots()
    # Word cloud
    wordcloud = WordCloud(
        background_color="white",
        max_words=500,
        width=640,
        height=360,
        collocations=False,
        stopwords=stopwords,
    ).generate(total_text)
    ax2.imshow(wordcloud, interpolation="bilinear")
    ax2.axis("off")
    ax2.set_title(
        f"Mots les plus fréquents sur prdtypecode: {code} avec traitement: {preprocessing}"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.write(fig2)

    col1, col2 = st.columns(2)

    # Bar plot
    n_words = 10
    fig3, ax3 = plt.subplots(figsize=(3, 2))
    ax3.bar(
        list(zip(*word_counts))[0][:n_words],
        list(zip(*word_counts))[1][:n_words],
    )
    ax3.tick_params(axis="x", rotation=90)
    col1.write(fig3)

    col2.dataframe(code_df[col_target].head(), width=1000)


# SIDEBAR
pages = [
    "textes",
    "images",
]
page = st.sidebar.radio("Explorer les données :", pages)

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
st.sidebar.progress(2 / 5)

# MAIN PAGE
st.title(f":blue[ANALYSE EXPLORATOIRE]")
st.write(f"## Les données : `{page}`")

if "X_train_df" not in st.session_state:
    st.session_state.X_train_df = pd.read_csv(
        f"{ROOT}data/raw/x_train.csv", index_col=0
    )
if "y_train_df" not in st.session_state:
    st.session_state.y_train_df = pd.read_csv(
        f"{ROOT}data/raw/y_train.csv", index_col=0
    )
if "X_train_preprocessed_df" not in st.session_state:
    st.session_state.X_train_preprocessed_df = pd.read_csv(
        f"{ROOT}data/processed/X_train_preprocessed.csv", index_col=0
    )
if "stop_words" not in st.session_state:
    stop_words_french = pd.read_json(ROOT + "data/external/stop_words_french.json")
    stop_words_english = pd.read_json(ROOT + "data/external/stop_words_english.json")
    stop_words = []
    stop_words.extend(stop_words_french[0].tolist())
    stop_words.extend(stop_words_english[0].tolist())
    stop_words.extend(["cm", "mm"])
    st.session_state.stop_words = stop_words

if page == pages[0]:
    df = st.session_state.X_train_df.copy()
    df["text"] = np.where(
        df["description"].isna(),
        df["designation"].astype(str),
        df["designation"].astype(str) + " " + df["description"].astype(str),
    )
    fig1, ax = plt.subplots(figsize=(10, 3))
    ax.boxplot(df["text"].str.len(), vert=False)
    ax.set_xlabel("Nombre de charactères")
    ax.set_title(f"Distribution du nombre de charactères sur la totalité")
    st.write(fig1)

    preprocessing = st.radio(
        "Prétraitement des données :",
        [
            "0 - Sans",
            "1 - Ponctuation, décodage HTML",
            "2 - Traduction, regex charactères spéciaux",
            "3 - Lemmatisation",
        ],
        horizontal=True,
    )

    code = st.selectbox(
        "Choisir le type de produits :",
        st.session_state.y_train_df["prdtypecode"].unique(),
        placeholder="Choisir le type de produits :",
    )

    create_word_cloud(preprocessing, st.session_state.y_train_df, code)

else:
    st.write("les données images")
