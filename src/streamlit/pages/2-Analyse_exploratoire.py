import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import random
from PIL import Image
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
    elif preprocessing == "2 - Traduction, regex charactères spéciaux, stopwords":
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

    fig, axes = plt.subplots(1, 4, figsize=(15, 10))

    # Afficher l'image d'origine
    axes[0].imshow(image_rgb)
    axes[0].set_title("Origine\n({0} pixels)".format(nb_pixels_avant))
    axes[0].axis("off")

    # Afficher l'image segmentée
    axes[1].imshow(result_rgb)
    axes[1].set_title("Image Segmentée\n({0} pixels)".format(nb_pixels_apres))
    axes[1].axis("off")

    # Afficher l'image redimensionnée
    axes[2].imshow(crop_rgb)
    axes[2].set_title("Image Détourée\n({0} pixels)".format(nb_pixels_apres2))
    axes[2].axis("off")

    # Afficher l'image zoomée
    axes[3].imshow(image_redim_rgb)
    axes[3].set_title("Image Redimensionnée\n({0} pixels)".format(nb_pixels_apres3))
    axes[3].axis("off")

    st.write(fig)


def appliquer_filtre(data, filtre):
    # Appliquer le filtre sélectionné et retourner le résultat
    path = f"{ROOT}data/raw/images/image_train/image_{data['imageid']}_product_{data['productid']}.jpg"
    # Charger l'image en niveaux de gris
    image_a_filtrer = cv2.imread(path)
    gray = cv2.cvtColor(image_a_filtrer, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if filtre == "Filtre de Sobel":
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        image_filtree = cv2.cvtColor(cv2.convertScaleAbs(sobel), cv2.COLOR_GRAY2RGB)
    elif filtre == "Filtre de Roberts":
        roberts_cross_v = cv2.filter2D(blurred, -1, np.array([[-1, 0], [0, 1]]))
        roberts_cross_h = cv2.filter2D(blurred, -1, np.array([[0, -1], [1, 0]]))
        roberts_cross = cv2.addWeighted(roberts_cross_v, 0.5, roberts_cross_h, 0.5, 0)
        image_filtree = cv2.cvtColor(
            cv2.convertScaleAbs(roberts_cross), cv2.COLOR_GRAY2RGB
        )
    elif filtre == "Filtre de Prewitt":
        kernelx = cv2.getDerivKernels(1, 0, 3)
        kernely = cv2.getDerivKernels(0, 1, 3)
        prewitt_x = cv2.filter2D(blurred, -1, kernelx[0] * kernelx[1].T)
        prewitt_y = cv2.filter2D(blurred, -1, kernely[0] * kernely[1].T)
        prewitt_cross = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
        image_filtree = cv2.cvtColor(
            cv2.convertScaleAbs(prewitt_cross), cv2.COLOR_GRAY2RGB
        )
    elif filtre == "Filtre de Laplace":
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        image_filtree = cv2.cvtColor(cv2.convertScaleAbs(laplacian), cv2.COLOR_GRAY2RGB)
    elif filtre == "Filtre de Canny":
        edges = cv2.Canny(blurred, 70, 255)
        image_filtree = cv2.cvtColor(cv2.convertScaleAbs(edges), cv2.COLOR_GRAY2RGB)
    elif filtre == "Seuil adaptatif":
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        image_filtree = cv2.cvtColor(
            cv2.convertScaleAbs(adaptive_thresh), cv2.COLOR_GRAY2RGB
        )
    else:
        image_filtree = cv2.cvtColor(image_a_filtrer, cv2.COLOR_BGR2RGB)
    return image_filtree


def afficher_images_en_ligne(images, nbcol):
    """Afficher les images

    Args:
        images (_type_): _description_
        nbcol (_type_): _description_
    """
    img_count = len(images)
    row_count = img_count // nbcol + (
        1 if img_count % nbcol != 0 else 0
    )  # Calcul du nombre de lignes
    for i in range(row_count):
        cols = st.columns(nbcol)  # Création de colonnes
        for j in range(nbcol):
            index = i * nbcol + j
            if index < img_count:
                chemin_image = os.path.join(chemin_images, images[index])
                cols[j].image(
                    chemin_image, width=100, use_column_width=False
                )  # Affichage de l'image dans la colonne avec une largeur de 100 pixels


# SIDEBAR
pages = ["Features : textes", "Features : images", "Target : Classe de Produits"]
page = st.sidebar.radio("Explorer les données :", pages)

# Pour descendre la suite
st.sidebar.markdown(
    """
#### 
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
    st.write("### Prétraitement des données")

    preprocessing = st.radio(
        "Prétraitement des données :",
        [
            "0 - Sans",
            "1 - Ponctuation, décodage HTML",
            "2 - Traduction, regex charactères spéciaux, stopwords",
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

elif page == pages[1]:
    chemin_images = f"{ROOT}data/raw/images/image_train"

    # Nombre d'images à afficher
    nombre_images_a_afficher = 32
    images_selectionnees = random.sample(
        os.listdir(chemin_images), nombre_images_a_afficher
    )

    # Affichage des images en ligne avec une taille réduite et des polices de taille réduite
    afficher_images_en_ligne(images_selectionnees, 8)
    st.write("Proportion d'images 500x500 : :blue[**98.924%**]")
    st.write(
        "La quasi totalité des images comporte une dimension de 500 x 500 alors qu'une observation d'un échantillon d'images ne le laisse pas supposer. Les images semblent donc avoir des :blue[**contours blancs**]."
    )
    st.write(
        "Il est donc nécessaire de :blue[**détourer les zones d'intérêt**] des images et les :blue[**redimensionner**]. Pour cela, nous allons utiliser plusieurs filtres pour définir les bords des images."
    )

    i = st.slider(
        "Selectionner l'index du produit :",
        0,
        st.session_state.X_train_df.shape[0] - 1,
        0,
        1,
    )

    # Appliquer le filtre sélectionné et afficher le résultat
    image_originale = appliquer_filtre(
        st.session_state.X_train_df.iloc[i - 1], "Origin"
    )
    image_filtree_sobel = appliquer_filtre(
        st.session_state.X_train_df.iloc[i - 1], "Filtre de Sobel"
    )
    image_filtree_rober = appliquer_filtre(
        st.session_state.X_train_df.iloc[i - 1], "Filtre de Roberts"
    )
    image_filtree_prewi = appliquer_filtre(
        st.session_state.X_train_df.iloc[i - 1], "Filtre de Prewitt"
    )
    image_filtree_lapla = appliquer_filtre(
        st.session_state.X_train_df.iloc[i - 1], "Filtre de Laplace"
    )
    image_filtree_canny = appliquer_filtre(
        st.session_state.X_train_df.iloc[i - 1], "Filtre de Canny"
    )
    image_filtree_adapt = appliquer_filtre(
        st.session_state.X_train_df.iloc[i - 1], "Seuil adaptatif"
    )

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.image(image_originale, caption="Image originale", use_column_width=True)
    col2.image(image_filtree_sobel, caption="Filtre de Sobel", use_column_width=True)
    col3.image(image_filtree_rober, caption="Filtre de Roberts", use_column_width=True)
    col4.image(image_filtree_prewi, caption="Filtre de Prewitt", use_column_width=True)
    col5.image(image_filtree_lapla, caption="Filtre de Laplace", use_column_width=True)
    col6.image(image_filtree_canny, caption="Filtre de Canny", use_column_width=True)
    col7.image(image_filtree_adapt, caption="Seuil adaptatif", use_column_width=True)

    st.write(
        "Tous les filtres testés nous permettent non seulement d':blue[**évincer les bords blancs**] de nos images mais également d':blue[**évincer l'arrière plan**] des images et donc de :blue[**concentrer l'analyse sur le premier plan**]."
    )
    st.write(
        "Après de nombreuses expérimentations nous avons choisi d'utiliser le filtre de :blue[**Prewitt**]. Après avoir détouré les images, nous les redimensionnons en 500 x 500 en étirant si besoin l'image détourée."
    )

    crop_the_image(st.session_state.X_train_df.iloc[i - 1])

elif page == pages[2]:
    target = st.session_state.y_train_df["prdtypecode"].astype("str")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    target.value_counts(sort=True).plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Type de produit")
    ax1.set_ylabel("Nombre de produits")
    ax1.set_title("Répartition des types de produit")
    st.write(fig1)
    target = st.session_state.y_train_df["prdtypecode"].astype("str")
    fig11, ax11 = plt.subplots(figsize=(10, 4))
    target.value_counts(sort=True, normalize=True).plot(kind="bar", ax=ax11)
    ax11.set_xlabel("Type de produit")
    ax11.set_ylabel("Pourcentage")
    ax11.set_title("Répartition des types de produit")
    st.write(fig11)

    st.write(
        "La Target comporte une distribution en 27 classes :blue[**fortement deséquilibrée**] avec certaines classes comportant plus de 10 fois moins d'observations que d'autres."
    )
    st.write(
        "Cette constatation va nous aiguiller sur des :blue[**techniques de reéchantillonnage**] (undersampling, oversampling SMOTE, datageneration pour les images) afin d'éviter que nos modèles connaissent des phénomènes d':blue[**overfitting**]. Pour la même raison le :blue[**Weigthed F1 Score**] sera une bonne métrique d'évaluation des modèles afin d'éviter de sur considérer les classes les plus représentées."
    )
