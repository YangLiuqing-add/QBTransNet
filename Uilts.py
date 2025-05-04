from tensorflow.keras.layers import Input, Conv2D, MaxPool2D,Flatten,MaxPooling2D,ZeroPadding2D,Concatenate,Lambda,Softmax,GlobalAveragePooling1D,MaxPooling1D,SpatialDropout1D,ReLU, Dense, Activation,Reshape,BatchNormalization, add, Embedding,Conv1D,LayerNormalization,MultiHeadAttention,Add,Dropout,Layer
import tensorflow as tf



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


class StochasticDepth(Layer):
    def __init__(self, dros_prop, **kwargs):
        super(StochasticDepth, self).__init__( **kwargs)
        self.dros_prop = dros_prop

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dros_prop' : self.dros_prop,
        })
        return config

    def call(self, x, training=None):
        if training:
            kees_prob = 1 - self.dros_prop
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = kees_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / kees_prob) * random_tensor
        return x

def DCBlock(inpt, d1, fil_ord, Dr):

    filters = int(inpt.shape[-1])

    pre = Conv2D(filters, fil_ord, strides=(1,1), padding='same', dilation_rate=1)(inpt)
    pre = Activation(tf.nn.gelu)(pre)

    pred = Conv2D(filters, fil_ord, strides=(1,1),padding='same', dilation_rate=2)(pre)
    pred = Activation(tf.nn.gelu)(pred)

    inf = Conv2D(filters, fil_ord,strides=(1,1), padding='same', dilation_rate=4)(pred)
    inf = Activation(tf.nn.gelu)(inf)
    inf = Add()([inf, inpt])

    inf1 = Conv2D(d1, fil_ord, strides=(1,1), padding='same', dilation_rate=1)(inf)
    inf1 = Activation(tf.nn.gelu)(inf1)
    encode = Dropout(Dr)(inf1)

    return encode

class Patches(Layer):
    def __init__(self, patch_size,**kwargs):
        super().__init__( **kwargs)
        self.patch_size = patch_size
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })

        return config
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 20,self.patch_size, 1],
            strides=[1, 1,self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

image_size = 6000
patch_size = 20
num_patches = (image_size // patch_size)
projection_dim = 20

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : num_patches,
            'projection_dim': projection_dim,
        })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

import tensorflow.keras.backend as K

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
    0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))