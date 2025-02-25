import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import dash
from dash import Input, Output, dcc, html
import io
import matplotlib.pyplot as plt
import base64
import plotly.graph_objects as go

def exponentiasl_decay_fn(epoch, current_lr):
    return current_lr * 0.3 ** (epoch / 10)

def augment_data(images, labels, augmentations=2):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,  # Mniejsze przesunięcia, by nie tracić istotnych cech
        height_shift_range=0.05,
        shear_range=0.05,  # Mniejsza transformacja kątowa
        zoom_range=0.1,  # Delikatne powiększenie
        horizontal_flip=False,  # Fashion MNIST nie zawiera obiektów, które powinny być lustrzane
        fill_mode='constant', cval=0  # Ustawienie czarnych pikseli w tle, by uniknąć artefaktów
    )
    augmented_images = []
    augmented_labels = []
    for i in range(len(images)):
        img = images[i].reshape((1, 28, 28, 1))
        label = labels[i]
        for _ in range(augmentations):
            aug_img = datagen.random_transform(img[0])  
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=10, stratify=train_labels
)
X_train = np.expand_dims(X_train, axis=-1).astype(np.float32)
X_test = np.expand_dims(X_test, axis=-1).astype(np.float32)

# Augmentacja danych
X_aug, y_aug = augment_data(X_train, y_train)
X_train_augmented = np.concatenate((X_train, X_aug), axis=0)
y_train_augmented = np.concatenate((y_train, y_aug), axis=0)

original_model_path = "fashion_mnist_model.h5"
augmented_model_path = "fashion_mnist_augmented_model_2.h5"

# Wczytanie lub trenowanie modelu podstawowego
if os.path.exists(original_model_path):
    model = load_model(original_model_path, custom_objects={"LeakyReLU": LeakyReLU})
    model.evaluate(X_test, y_test)
    print("Model podstawowy wczytany z pliku.")
else:
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=False),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1,1)),
        Dropout(0.2),

        Conv2D(64, (3, 3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=False),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1,1)),
        Dropout(0.3),

        Conv2D(128, (3, 3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=False),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1,1)),
        Dropout(0.4),

        Flatten(),
        Dense(128),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),  
        Dense(10, activation='softmax')  
    ])


    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.fit(X_train, 
              y_train, 
              epochs=10, 
              verbose=1, 
              callbacks=[tf.keras.callbacks.LearningRateScheduler(exponentiasl_decay_fn)],
              validation_split = 0.1)
    model.save(original_model_path)
    print("Model podstawowy zapisany do pliku.")

# Wczytanie lub trenowanie modelu na powiększonym zbiorze danych
if os.path.exists(augmented_model_path):
    model_augmented = load_model(augmented_model_path, custom_objects={"LeakyReLU": LeakyReLU})
    model_augmented.evaluate(X_test, y_test)
    print("Model z augmentacją wczytany z pliku.")
else:
    model_augmented = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=False),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1,1)),
        Dropout(0.2),

        Conv2D(64, (3, 3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=False),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1,1)),
        Dropout(0.3),

        Conv2D(128, (3, 3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1), use_bias=False),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1,1)),
        Dropout(0.4),

        Flatten(),
        Dense(128),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),  
        Dense(10, activation='softmax')  
    ])

    model_augmented.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    model_augmented.fit(X_train_augmented, 
                        y_train_augmented, 
                        epochs=10, 
                        verbose=1, 
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(exponentiasl_decay_fn)],
                        validation_split = 0.1)
    model_augmented.save(augmented_model_path)
    print("Model z augmentacją zapisany do pliku.")


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Inicjalizacja aplikacji Dash
app = dash.Dash(__name__)
server = app.server

test_loss_base, test_acc_base = model.evaluate(X_test, y_test, verbose=0)
train_loss_base, train_acc_base = model.evaluate(X_train, y_train, verbose=0)
test_loss_aug, test_acc_aug = model_augmented.evaluate(X_test, y_test, verbose=0)
train_loss_aug, train_acc_aug = model_augmented.evaluate(X_train, y_train, verbose=0)

app.layout = html.Div([
    html.H1("Wizualizacja predykcji", style={'text-align': 'center'}),
    html.H2("Wybierz indeks obrazu z Fashion-MNIST (0-9999)", style={'text-align': 'center'}),
    html.Div([
        dcc.Input(id='index-input', type='number', min=0, max=9999, value=0),
        html.Button('Pokaż', id='submit-button', n_clicks=0)
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.Img(id='output-image', style={'width': '300px', 'height': '300px'}),
            html.H3(id='true-label', style={'text-align': 'center'})
        ], style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
        html.Div([
            dcc.Graph(id='output-graph')
        ], style={'width': '50%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    html.Div([
        html.H3("Porównanie dokładności modeli", style={'text-align': 'center'}),
        html.Table([
            html.Tr([html.Th("Model"), html.Th("Accuracy na zbiorze testowym"), html.Th("Accuracy na zbiorze treningowym")]),
            html.Tr([html.Td("Model podstawowy"), html.Td(f"{test_acc_base:.4f}" if test_acc_base is not None else "Brak danych"), html.Td(f"{train_acc_base:.4f}" if train_acc_base is not None else "Brak danych")]),
            html.Tr([html.Td("Model po augmentacji"), html.Td(f"{test_acc_aug:.4f}" if test_acc_aug is not None else "Brak danych"), html.Td(f"{train_acc_aug:.4f}" if train_acc_aug is not None else "Brak danych")])
        ], style={'margin': 'auto', 'text-align': 'center', 'border': '1px solid black', 'border-collapse': 'collapse', 'width': '50%'})
    ], style={'width': '100%', 'text-align': 'center'})
])

@app.callback(
    [Output('output-image', 'src'), Output('output-graph', 'figure'), Output('true-label', 'children')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('index-input', 'value')]
)
def update_output(n_clicks, index):
    if index is None or index < 0 or index >= len(X_test):
        return None, go.Figure()

    image = X_test[index]
    image_reshaped = np.expand_dims(image, axis=(0, -1))
    
    predictions = model_augmented.predict(image_reshaped)
    prediction = predictions[0]
    predicted_label = np.argmax(prediction)

    fig = go.Figure([go.Bar(x=prediction, y=class_names, orientation='h', marker_color='blue')])
    fig.update_layout(title_text=f'Predykcja: {class_names[predicted_label]} ({100 * np.max(prediction):.2f}%)',
                      xaxis_title='Prawdopodobieństwo', yaxis_title='Klasy', yaxis=dict(autorange='reversed'))

    buf = io.BytesIO()
    plt.imsave(buf, image.squeeze(), cmap='gray')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    img_src = f'data:image/png;base64,{encoded_image}'

    true_label_text = f'Prawdziwa klasa: {class_names[int(y_test[index])]}'
    return img_src, fig, true_label_text

if __name__ == '__main__':
    app.run_server(debug=True)