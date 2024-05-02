import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import cv2

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


def crop_the_image(data):
    path = f"{ROOT}data/raw/images/image_train/image_{data['imageid']}_product_{data['productid']}.jpg"
    # Charger l'image en niveaux de gris
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    kernelx = cv2.getDerivKernels(1, 0, 3)
    kernely = cv2.getDerivKernels(0, 1, 3)
    prewitt_x = cv2.filter2D(blurred, -1, kernelx[0] * kernelx[1].T)
    prewitt_y = cv2.filter2D(blurred, -1, kernely[0] * kernely[1].T)
    prewitt_cross = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

    contours, _ = cv2.findContours(
        prewitt_cross, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1
    )

    bounding_boxes = []
    for contour in contours:
        # Trouver les coordonnées de la boîte englobante du contour actuel
        x, y, w, h = cv2.boundingRect(contour)
        # Ajouter les coordonnées de la boîte englobante à la liste
        bounding_boxes.append((x, y, w, h))

    bounding_boxes_array = np.array(bounding_boxes)
    # Utiliser la fonction min() et max() de numpy pour obtenir les valeurs minimales et maximales
    min_x, min_y, min_w, min_h = np.min(bounding_boxes_array, axis=0)
    max_x, max_y, max_w, max_h = np.max(bounding_boxes_array, axis=0)

    # Extraire la région de l'image couverte par la boîte englobante
    cropped_image = image[min_y : min_y + max_h, min_x : min_x + max_w]
    if cropped_image.size == 0:
        print("L'image cropped_image est vide.")
    else:
        # Convertir l'image en niveaux de gris
        gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Redimensionner l'image en conservant le ratio d'aspect
    image_redim = cv2.resize(cropped_image, (500, 500))

    # Obtenir les dimensions d'origine de l'image cropped_image
    hauteur_origine, largeur_origine = cropped_image.shape[:2]

    # Calculer le ratio de redimensionnement
    ratio = min(500 / largeur_origine, 500 / hauteur_origine)

    # Appliquer le ratio pour redimensionner l'image tout en conservant les proportions
    image_redim = cv2.resize(cropped_image, None, fx=ratio, fy=ratio)

    # Créer une nouvelle image blanche de taille 500x500
    image_redim2 = np.zeros((500, 500, 3), dtype=np.uint8)
    image_redim2.fill(255)  # Remplir l'image avec du blanc (255)

    # Calculer les coordonnées pour placer l'image redimensionnée au centre de la nouvelle image
    x_offset = (500 - image_redim.shape[1]) // 2
    y_offset = (500 - image_redim.shape[0]) // 2

    # Placer l'image redimensionnée au centre de la nouvelle image
    image_redim2[
        y_offset : y_offset + image_redim.shape[0],
        x_offset : x_offset + image_redim.shape[1],
    ] = image_redim

    image_redim = cv2.resize(cropped_image, (500, 500))
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    gray_segmented_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray_image_redim = cv2.cvtColor(image_redim, cv2.COLOR_BGR2GRAY)
    gray_image_redim2 = cv2.cvtColor(image_redim2, cv2.COLOR_BGR2GRAY)

    nb_pixels_avant = image.shape[0] * image.shape[1]
    nb_pixels_apres = gray_segmented_image.shape[0] * gray_segmented_image.shape[1]
    nb_pixels_apres2 = gray_cropped_image.shape[0] * gray_cropped_image.shape[1]
    nb_pixels_apres3 = gray_image_redim.shape[0] * gray_image_redim.shape[1]
    nb_pixels_apres4 = gray_image_redim2.shape[0] * gray_image_redim2.shape[1]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    image_redim_rgb = cv2.cvtColor(image_redim, cv2.COLOR_BGR2RGB)
    image_redim2_rgb = cv2.cvtColor(image_redim2, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(15, 10))

    # Afficher l'image d'origine
    axes[0].imshow(image_rgb)
    axes[0].set_title("Origine\n({0} pixels)".format(nb_pixels_avant))
    axes[0].axis("off")

    # Afficher l'image segmentée
    axes[1].imshow(result_rgb)
    axes[1].set_title("Segmentation\n({0} pixels)".format(nb_pixels_apres))
    axes[1].axis("off")

    # Afficher l'image redimensionnée
    axes[2].imshow(crop_rgb)
    axes[2].set_title("Crop\n({0} pixels)".format(nb_pixels_apres2))
    axes[2].axis("off")

    # Afficher l'image zoomée
    axes[3].imshow(image_redim_rgb)
    axes[3].set_title("Zoom ss prop\n({0} pixels)".format(nb_pixels_apres3))
    axes[3].axis("off")

    # Afficher l'image zoomée en conservant les proportions
    axes[4].imshow(image_redim2_rgb)
    axes[4].set_title("Zoom avec prop\n({0} pixels)".format(nb_pixels_apres4))
    axes[4].axis("off")

    st.write(fig)


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
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.boxplot(df["text"].str.len(), vert=False)
    ax.set_xlabel("Nombre de charactères")
    ax.set_title(f"Distribution du nombre de charactères sur designation + description")
    st.write(fig)

    col1, col2 = st.columns(2)

    target = st.session_state.y_train_df["prdtypecode"].astype("str")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    target.value_counts(sort=True).plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Type de produit")
    ax1.set_ylabel("Nombre de produits")
    ax1.set_title("Répartition des types de produit")
    col1.write(fig1)
    target = st.session_state.y_train_df["prdtypecode"].astype("str")
    fig11, ax11 = plt.subplots(figsize=(10, 4))
    target.value_counts(sort=True, normalize=True).plot(kind="bar", ax=ax11)
    ax11.set_xlabel("Type de produit")
    ax11.set_ylabel("Pourcentage")
    ax11.set_title("Répartition des types de produit")
    col2.write(fig11)

    st.write("### Prétraitement des données")

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
    st.write(
        """Après avoir analysé les images du dataset, il a été constaté que la majorité des images ont une taille de :blue[**500 x 500 pixels**] (98,9% du dataset). Il semble que :blue[**Rakuten ait normalisé les images**] en ajoutant un contour blanc pour atteindre la taille de 500 x 500 pixels."""
    )
    st.write(
        """Une opération de :blue[**détourage (cropping)**] a été réalisée pour supprimer les contours blancs et les fonds d'image sans intérêt. Après le cropping, les tailles réelles des images étaient différentes des 500 x 500 initiaux."""
    )
    i = st.slider(
        "Selectionner l'index du produit :",
        0,
        st.session_state.X_train_df.shape[0],
        0,
        1,
    )
    crop_the_image(st.session_state.X_train_df.iloc[i])
    st.image(
        f"{ROOT}reports/tailles_images_apres_detourage.png",
        caption="Dimension des images après détourage (Hauteur x Largeur)",
        use_column_width=True,
    )
    st.image(
        f"{ROOT}reports/dist_n_pixels_apres_detourage_prdtypecode.png",
        caption="Distribution du nombre de pixels après détourage par type de produit",
        use_column_width=True,
    )
