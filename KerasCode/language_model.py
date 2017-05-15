from keras.models import Model
from keras.layers.recurrent import *
from keras.layers.embeddings import *
from keras.layers import *
from keras.optimizers import *

embed_dim = 200
vocab_size = 3
num_neg = 1
sen_len = 5
word_embed_data = np.random.rand(vocab_size, embed_dim)


words_input_pos = Input(shape=(sen_len,), dtype="int32", name="words_input_pos")
words_input_neg = Input(shape=(sen_len,), dtype="int32", name="words_input_neg")

def get_sim(x):
    return K.batch_dot(x[0], x[1], axes=2)

def merge_sim(x):
    return K.concatenate(x, axis=2)

word_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=True, name="word_layer",
                       weights=[word_embed_data])
lstm_layer = LSTM(embed_dim, return_sequences=True, name="lstm_layer")
sim_layer = Lambda(function=get_sim, name="sim_layer")
merge_layer = Lambda(function=merge_sim, name="merge_layer")

words_embed_pos = word_layer(words_input_pos)
words_embed_neg = word_layer(words_input_neg)
lstm_embed = lstm_layer(words_embed_pos)
pos_sim = sim_layer([lstm_embed, words_embed_pos])
neg_sim = sim_layer([lstm_embed, words_embed_neg])
merge_embed = merge_layer([pos_sim, neg_sim])


def hinge_loss(y_true, y_pred):
    loss = sen_len
    for i in range(sen_len):
        loss -= y_pred[:, i, i]
        loss += y_pred[:, i, i+sen_len]
    loss = K.mean(K.maximum(loss, 0.0))
    return loss


model = Model(input=[words_input_pos, words_input_neg], output=[merge_embed])
model.compile(optimizer=Adam(lr=0.001), loss = hinge_loss)
print(model.summary())

data_pos = np.ones((10, sen_len))
data_neg = np.ones((10, sen_len))

pseudo_output = np.ones((10, sen_len, 2))

model.fit({"words_input_pos":data_pos, "words_input_neg": data_neg}, {"merge_layer":pseudo_output}, batch_size=1, nb_epoch=1)


