from sklearn.calibration import LabelEncoder
import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import re
import string
from tensorflow.keras.layers import TextVectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from skimage import io, color, feature, transform
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical

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
    options_models = ["MultinomialNB"]
elif type_data == "Image":
    options_models = ["RF_HOG"]
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
    preds_df = pd.DataFrame(
        loaded_model.predict_proba(data), columns=loaded_model.classes_
    )
    code = int(preds_df.idxmax(axis=1)[0])
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
    preds_df = pd.DataFrame(
        loaded_model.predict_proba(data), columns=loaded_model.classes_
    )
    code = int(preds_df.idxmax(axis=1)[0])
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
    preds_df = pd.DataFrame(
        loaded_model.predict_proba(data), columns=loaded_model.classes_
    )
    code = int(preds_df.idxmax(axis=1)[0])
elif modele == "Stacking":
    resize = (224, 224)
    sequence_length = 1000

    def dataframe_to_dataset(dataframe):
        columns = ["image_path", "text"]
        dataframe = dataframe[columns].copy()
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
        return ds

    def read_resize(image_path):
        extension = tf.strings.split(image_path)[-1]

        image = tf.io.read_file(image_path)
        if extension == b"jpg":
            image = tf.image.decode_jpeg(image, 3)
        else:
            image = tf.image.decode_png(image, 3)
        image = tf.image.resize(image, resize)
        return image

    def custom_standardization(input_data):
        """
        Custom standardization function for text data.

        Args:
            input_data: The input text data.

        Returns:
            The standardized text data.
        """
        decoded_html = tf.strings.unicode_decode(input_data, "UTF-8")
        encoded_html = tf.strings.unicode_encode(decoded_html, "UTF-8")
        stripped_html = tf.strings.regex_replace(encoded_html, "<[^>]*>", " ")
        lowercase = tf.strings.lower(stripped_html)
        cleaned_input_data = tf.strings.regex_replace(lowercase, r"\s+", " ")
        print(cleaned_input_data)
        return tf.strings.regex_replace(
            cleaned_input_data, "[%s]" % re.escape(string.punctuation), ""
        )

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=100000,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    df = st.session_state.X_train_df.copy()
    df["text"] = np.where(
        df["description"].isna(),
        df["designation"].astype(str),
        df["designation"].astype(str) + " " + df["description"].astype(str),
    )
    vectorize_layer.adapt(df["text"])

    def preprocess_text(text):
        text = vectorize_layer(text)
        text = tf.convert_to_tensor(text)
        return text

    def preprocess(sample):
        image = read_resize(sample["image_path"])
        text = preprocess_text(sample["text"])
        return {"image": image, "text": text}

    batch_size = 32 * 2
    auto = tf.data.AUTOTUNE

    def prepare_dataset(df):
        ds = dataframe_to_dataset(df)
        ds = ds.map(lambda x: (preprocess(x)))
        ds = ds.batch(batch_size).prefetch(auto)
        return ds

    data["image_name"] = data.apply(
        lambda row: f"image_{row['imageid']}_product_{row['productid']}.jpg", axis=1
    )
    data["image_path"] = (
        ROOT + "data/raw/images/image_test/" + data["image_name"].astype("str")
    )
    data = prepare_dataset(data)

    logged_model = "runs:/efbc34a6fe754c53954d428d763047a8/model"
    loaded_model = mlflow.tensorflow.load_model(logged_model)
    classes = np.sort(st.session_state.y_train_df["prdtypecode"].unique())
    preds_df = pd.DataFrame(loaded_model.predict(data), columns=classes)

    code = int(preds_df.idxmax(axis=1)[0])

prdtypecodes = {
    10: "10 - Livres occasion",
    40: "40 - Jeux vidéos et consoles neufs",
    50: "50 - Accessoires gaming",
    60: "60 - Consoles de jeux occasion",
    1140: "1140 - Objets pop culture",
    1160: "1160 - Cartes de jeu",
    1180: "1180 - Jeux de rôle et figurines",
    1280: "1280 - Jouets enfant",
    1281: "1281 - Jeux enfants",
    1300: "1300 - Modélisme",
    1301: "1301 - Chaussettes enfants",
    1302: "1302 - Jeux de plein air",
    1320: "1320 - Puériculture",
    1560: "1560 - Mobilier",
    1920: "1920 - Linge de maison",
    1940: "1940 - Épicerie",  # Modification ici
    2060: "2060 - Décoration",
    2220: "2220 - Animalerie",
    2280: "2280 - Journaux / magazines occasion",
    2403: "2403 - Lots livres et magazines",
    2462: "2462 - Jeux vidéo occasion",
    2522: "2522 - Fournitures, papeterie",
    2582: "2582 - Mobilier de jardin",
    2583: "2583 - Piscine et accessoires",
    2585: "2585 - Outillage de jardin",
    2705: "2705 - Livres neufs",
    2905: "2905 - Jeux PC",
}
st.write("**Prédiction du prdtypecode :**")
st.success(prdtypecodes[code])
st.write("**Probabilités de prédiction des prdtypecodes :**")
st.write(preds_df)
create_word_cloud(code)
