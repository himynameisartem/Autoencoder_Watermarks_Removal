from keras.models import Model
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Input,
    UpSampling2D,
    BatchNormalization,
    Concatenate,
    Resizing,
    LeakyReLU
)
from keras.optimizers import Adam


def create_unet_autoencoder(img_x: int, img_y: int, channel: int, *, base_filters: int = 32):

    def conv_block(x, filters: int):
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        return x

    inputs = Input(shape=(img_x, img_y, channel))

    e1 = conv_block(inputs, base_filters)
    p1 = MaxPooling2D((2, 2), padding="same")(e1)

    e2 = conv_block(p1, base_filters * 2)
    p2 = MaxPooling2D((2, 2), padding="same")(e2)

    b = conv_block(p2, base_filters * 4)

    u2 = UpSampling2D((2, 2))(b)
    u2 = Concatenate()([u2, e2])
    d2 = conv_block(u2, base_filters * 2)

    u1 = UpSampling2D((2, 2))(d2)
    u1 = Concatenate()([u1, e1])
    d1 = conv_block(u1, base_filters)

    outputs = Conv2D(channel, (1, 1), activation="sigmoid", padding="same")(d1)

    model = Model(inputs, outputs, name="unet_autoencoder")
    model.compile(optimizer=Adam(learning_rate=3e-4),
                  loss="mse", metrics=["mae"])
    return model


def create_light_unet(img_h, img_w, channels=3):
    inputs = Input(shape=(img_h, img_w, channels))

    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    b = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)

    u1 = UpSampling2D((2, 2))(b)
    u1 = Resizing(c2.shape[1], c2.shape[2])(u1)
    concat1 = Concatenate()([u1, c2])
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)

    u2 = UpSampling2D((2, 2))(c3)
    u2 = Resizing(c1.shape[1], c1.shape[2])(u2)
    concat2 = Concatenate()([u2, c1])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)

    outputs = Conv2D(channels, (3, 3), activation='sigmoid',
                     padding='same')(c4)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_model(img_h, img_w, channels=3):
    inputs = Input(shape=(img_h, img_w, channels))

    c1 = Conv2D(32, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU(alpha=0.1)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU(alpha=0.1)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    b = Conv2D(128, (3, 3), padding='same')(p2)
    b = BatchNormalization()(b)
    b = LeakyReLU(alpha=0.1)(b)

    u1 = UpSampling2D((2, 2))(b)
    u1 = Resizing(c2.shape[1], c2.shape[2])(u1)
    concat1 = Concatenate()([u1, c2])
    c3 = Conv2D(64, (3, 3), padding='same')(concat1)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=0.1)(c3)

    u2 = UpSampling2D((2, 2))(c3)
    u2 = Resizing(c1.shape[1], c1.shape[2])(u2)
    concat2 = Concatenate()([u2, c1])
    c4 = Conv2D(32, (3, 3), padding='same')(concat2)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU(alpha=0.1)(c4)

    outputs = Conv2D(channels, (3, 3), activation='sigmoid',
                     padding='same')(c4)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model
