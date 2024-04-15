import numpy as np
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    Bidirectional,
    Dropout,
    LSTM,
)
from keras import Sequential, losses, optimizers
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Constantes
SEED = 123


def run(useMlFlow: bool):
    if useMlFlow:
        mlflow.set_tracking_uri("mlruns")  # "http://127.0.0.1:8080"
        mlflow.set_experiment(experiment_name="RNN_model")
        mlflow.autolog(log_datasets=False)

    df = pd.read_csv("data/raw/x_train.csv", index_col=0)
    df_target = pd.read_csv("data/raw/y_train.csv", index_col=0)
    df[df_target.columns[0]] = df_target

    df["text"] = np.where(
        df["description"].isna(),
        df["designation"].astype(str),
        df["designation"].astype(str) + " " + df["description"].astype(str),
    )

    df.drop("designation", axis=1, inplace=True)
    df.drop("description", axis=1, inplace=True)
    df.drop("productid", axis=1, inplace=True)
    df.drop("imageid", axis=1, inplace=True)

    num_classes = df["prdtypecode"].value_counts().shape[0]

    data = df["text"]
    target = df["prdtypecode"].astype("str")

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.15, random_state=SEED
    )

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # Encode
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Vectorize
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(y_train.shape)

    y_train_1d = np.argmax(y_train, axis=1)

    # Appliquer des poids aux classes selon l'équilibrage du dataset
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_1d), y=y_train_1d
    )

    # Create a dictionary mapping class indices to their corresponding weights
    class_weight_dict = dict(zip(np.unique(y_train_1d), class_weights))

    # Défnit la longueur de la séquence du model
    # Vocabulary size and number of words in a sequence.
    # median = 320
    # mean = 600
    df["len"] = df["text"].str.len()
    sequence_length = int(df["len"].quantile(0.10))
    print("sequence_length:", sequence_length)

    # Pour libérer de la RAM
    del df, data, df_target, target

    X_train = tf.strings.as_string(X_train)
    X_test = tf.strings.as_string(X_test)

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=50000,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    vectorize_layer.adapt(X_train)
    print("type:", type(X_train), "  shape: ", X_train.shape)

    print(tf.config.list_physical_devices("GPU"))

    model = Sequential(
        [
            vectorize_layer,
            Embedding(len(vectorize_layer.get_vocabulary()), 64, mask_zero=True),
            Bidirectional(LSTM(31, return_sequences=True)),
            Bidirectional(LSTM(16)),
            Dense(32, activation="relu"),
            Dropout(0.4),
            Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile the model
    opt = optimizers.Adam(0.001)
    loss = losses.CategoricalCrossentropy()
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["accuracy"],
    )

    model.build((None, sequence_length))

    print(model.summary())

    # Callbacks
    early_stopping = EarlyStopping(
        patience=10,  # Attendre n epochs avant application
        min_delta=0.005,  # si au bout de n epochs la fonction de perte ne varie pas de n %,
        # que ce soit à la hausse ou à la baisse, on arrête
        verbose=1,  # Afficher à quel epoch on s'arrête
        mode="min",
        monitor="val_loss",
    )
    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_loss",
        patience=5,  # si val_loss stagne sur n epochs consécutives selon la valeur min_delta
        min_delta=0.005,
        factor=0.2,  # On réduit le learning rate d'un facteur n
        cooldown=3,  # On attend n epochs avant de réitérer
        verbose=1,
    )

    # Train the model
    training_history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[reduce_learning_rate, early_stopping],
        class_weight=class_weight_dict,
    )

    plt_graph(training_history)


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
    return tf.strings.regex_replace(
        cleaned_input_data, "[%s]" % re.escape(string.punctuation), ""
    )


def plt_graph(training_history):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    range_epochs = np.arange(1, len(training_history.epoch) + 1, 1)

    # Courbe de la précision sur l'échantillon d'entrainement
    ax1.plot(
        range_epochs,
        training_history.history["accuracy"],
        label="Training Accuracy",
        color="blue",
    )

    ax1.plot(
        range_epochs,
        training_history.history["val_accuracy"],
        label="Validation Accuracy",
        color="orange",
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.legend()

    # Courbe de la précision sur l'échantillon de test
    ax2.plot(
        range_epochs,
        training_history.history["loss"],
        label="Training Loss",
        color="blue",
    )
    ax2.plot(
        range_epochs,
        training_history.history["val_loss"],
        label="Validation Loss",
        color="orange",
    )
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and Validation Loss")
    ax2.legend()

    # Affichage de la figure
    plt.savefig("reports/figures/training_history_rnn_texts.png")


if __name__ == "__main__":
    # if yes, run before : mlflow server --host 0.0.0.0 --port 8080
    run(
        useMlFlow=input("Do you want to run the experiment with MlFlow? (y/n): ") == "y"
    )
