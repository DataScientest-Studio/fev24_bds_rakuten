import streamlit as st
import mlflow
import pandas as pd
import numpy as np
from tensorflow.keras.layers import TextVectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from skimage import io, color, feature, transform
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

ROOT = "../../"

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
""",
)
st.sidebar.progress(4 / 5)

if "X_test_df" not in st.session_state:
    st.session_state.X_test_df = pd.read_csv(f"{ROOT}data/raw/x_test.csv", index_col=0)
if "X_train_df" not in st.session_state:
    st.session_state.X_train_df = pd.read_csv(
        f"{ROOT}data/raw/x_train.csv", index_col=0
    )
if "y_train_df" not in st.session_state:
    st.session_state.y_train_df = pd.read_csv(
        f"{ROOT}data/raw/y_train.csv", index_col=0
    )
if "X_train_df_komla" not in st.session_state:
    st.session_state.X_train_df_komla = pd.read_csv(
        f"{ROOT}data/processed/X_train_update (komla).csv", index_col=0
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


def create_word_cloud(code):
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
    ).generate(total_text)
    ax2.imshow(wordcloud, interpolation="bilinear")
    ax2.axis("off")
    ax2.set_title(f"Mots les plus fréquents sur prdtypecode: {code}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.write(fig2)


# Fonction pour extraire les caractéristiques HOG d'une image
def extract_hog_features(image_path):
    image = io.imread(image_path)
    image_gray = color.rgb2gray(image)
    image_resized = transform.resize(image_gray, (128, 64), anti_aliasing=True)
    hog_features = feature.hog(
        image_resized, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False
    )
    return hog_features


def show_data(df):
    # Join texts
    df["text"] = np.where(
        df["description"].isna(),
        df["designation"].astype(str),
        df["designation"].astype(str) + " " + df["description"].astype(str),
    )
    col1, col2 = st.columns(2)
    col1.write(df["text"].values[0])
    col2.image(
        f"{ROOT}data/raw/images/image_test/image_{df['imageid'].values[0]}_product_{df['productid'].values[0]}.jpg"
    )
    return df


# MAIN PAGE
st.title(":blue[DÉMONSTRATION]")

type_data = st.selectbox(
    "Choix du type de données d'entrée :", ["Texte", "Image", "Texte & image"]
)
if type_data == "Texte":
    options_models = ["DNN", "MultinomialNB"]
elif type_data == "Image":
    options_models = ["CNN_EfficientNetB0", "RF_HOG"]
else:
    options_models = ["Stacking", "Features RF"]
modele = st.selectbox("Choix du modèle :", options_models)


# Predict on a Pandas DataFrame.
i = st.slider(
    "Selectionner l'index du produit :",
    0,
    st.session_state.X_test_df.index[-1] - st.session_state.X_test_df.index[0],
    0,
    1,
)

# Load model as a PyFuncModel.
mlflow.set_tracking_uri("../../mlruns")
data = show_data(st.session_state.X_test_df.iloc[i].to_frame().transpose())
if modele == "Features RF":
    # Configuration pour la vectorisation du texte
    vectorize_layer = TextVectorization(
        max_tokens=10000,
        output_mode="int",
        output_sequence_length=250,
    )
    vectorize_layer.adapt(st.session_state.X_train_df_komla["description"].fillna(""))
    text_vectorized = vectorize_layer(data["text"])
    # Préparation des caractéristiques HOG pour les images
    features_images = np.array(
        [
            extract_hog_features(
                f"{ROOT}data/raw/images/image_test/image_{imageid}_product_{productid}.jpg"
            )
            for imageid, productid in zip(data["imageid"], data["productid"])
        ]
    )
    data = np.hstack((features_images, text_vectorized))
    logged_model = "runs:/3ed6f6fd6971442db10cff63789ff786/model"
    loaded_model = mlflow.sklearn.load_model(logged_model)
elif modele == "MultinomialNB":
    df = st.session_state.X_train_preprocessed_df.copy()
    value_counts = df["prdtypecode"].value_counts()
    median = int(value_counts.median())
    value_counts_index = value_counts[value_counts > median].index
    index_row = []
    for i in value_counts.index:
        if i in value_counts_index:
            index_row.extend(
                df.loc[df["prdtypecode"] == i].sample(n=median, random_state=123).index
            )
        else:
            index_row.extend(df.loc[df["prdtypecode"] == i].index)
    df = df.loc[df.index.isin(index_row)]["text"]
    stop_words_french = pd.read_json(f"{ROOT}data/external/stop_words_french.json")
    stop_words = []
    stop_words.extend(stop_words_french[0].tolist())
    stop_words.extend(["cm", "mm"])
    tfidf_vect = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.5,
        min_df=0.00001,
        lowercase=True,
        stop_words=stop_words,
        max_features=350000,
    )
    tfidf_vect.fit(df)
    data = tfidf_vect.transform(data["text"])
    logged_model = "runs:/3cc7c2da8a064176a5622aa978ca3d65/best_estimator"
    loaded_model = mlflow.sklearn.load_model(logged_model)
elif modele == "RF_HOG":
    data = np.array(
        [
            extract_hog_features(
                f"{ROOT}data/raw/images/image_test/image_{imageid}_product_{productid}.jpg"
            )
            for imageid, productid in zip(data["imageid"], data["productid"])
        ]
    )
    logged_model = "runs:/d4a6f0783f7a441c82cea249802cde4d/best_estimator"
    loaded_model = mlflow.sklearn.load_model(logged_model)
else:
    logged_model = ""
preds_df = pd.DataFrame(loaded_model.predict_proba(data), columns=loaded_model.classes_)
code = int(preds_df.idxmax(axis=1)[0])
st.success(
    f"Prédiction du prdtypecode : {code}",
)
st.write(preds_df)
create_word_cloud(code)
