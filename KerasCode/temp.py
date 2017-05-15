from keras.models import Model
from keras.layers.recurrent import *
from keras.layers.embeddings import *
from keras.layers import *
from keras.optimizers import *
embed_dim = 20
lstm_dim = 10
vocab_size = 3
sen_len = 5
word_embed_data = np.random.rand(vocab_size, embed_dim)

word_input1 = Input(shape=(2,2), dtype="float32", name="word_input1")
word_input2 = Input(shape=(2,2), dtype="float32", name="word_input2")

def mmm(x):
    # return K.batch_dot(x[0], x[1], axes=(2,2))
    return x[0] +x[1]

merge_layer = Lambda(function=mmm, name="mmm")

merge_result = merge_layer([word_input1, word_input2])

def loss(y_true, y_pred):
    return y_true[:,1,0]


model = Model(input=[word_input1, word_input2], output=[merge_result])
model.compile(optimizer=Adam(lr=0.001), loss = loss)
print(model.summary())

data = np.asarray([[1,2],[3,4]])
data = np.asarray([data] * 3)
# y = model.predict({"word_input1":data, "word_input2":data}, batch_size=1, verbose=1)
# print(y)

y = model.evaluate({"word_input1":data, "word_input2":data},{"mmm": data}, batch_size=3, verbose=1)
print("\nloss = ",y)

