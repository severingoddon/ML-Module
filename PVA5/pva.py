from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.python.keras import regularizers

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_train  = X_train.shape[0]
num_test   = X_test.shape[0]

img_height = X_train.shape[1]
img_width  = X_train.shape[2]

X_train = X_train.reshape((num_train, img_width * img_height))
X_test  = X_test.reshape((num_test, img_width * img_height))

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

num_classes = 10
xi = Input(shape=(img_height*img_width,), name="input")

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1.0
xl = Lambda(lambda image, mu, std: (image - mu) / std,
           arguments={'mu': mean, 'std': std})(xi)
xo = Dense(num_classes, name="y",
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(xl)
yo = Activation('softmax', name="y_act")(xo)
model = Model(inputs=[xi], outputs=[yo])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])