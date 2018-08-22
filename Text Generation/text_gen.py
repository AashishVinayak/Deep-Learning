import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


# importing data
data = open('tom.txt').read()

# vocabulary
vocab = sorted(list(set(data)))
# character to index mapping
chr_to_id = {c:i for i, c in enumerate(vocab)}
# index to character mapping
id_to_chr = {i:c for i, c in enumerate(vocab)}


## converting data to list will seperate the characters
# 2 empty lists to hold input and output sequences
x_train = []
y_train = []
# converting text to list will separate all the characters
data = list(data)
# no. of characters present in one sequence training example
seq_len = 50
# training set
# loop over all the characters in the text till seq_len less the given size
for i in range(len(data[0:330172])-seq_len):
    # append the index of the ith to i + seq_len characters into one training example
    x_train.append( [chr_to_id[letter] for letter in  data[i:i+seq_len]] )
    # append the index of the character that comes right after the sequence in the above line
    y_train.append( chr_to_id[data[i+seq_len]] )

# test set
x_test = []
y_test = []
for i in range(len(data[330172:len(data)])-seq_len):
    # same process as above
    x_test.append( [chr_to_id[letter] for letter in  data[i:i+seq_len]] )
    y_test.append( chr_to_id[data[i+seq_len]] )

# reshape input data
x_train = np.array(x_train).reshape(len(x_train), seq_len, 1)
x_test = np.array(x_test).reshape(len(x_test), seq_len, 1)

# normalize input data
x_train = x_train/len(vocab)
x_test = x_test/len(vocab)

# categorizing output labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


###########    MODEL    ###########
model = Sequential()
model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2])), return_sequences = True)
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss = "categorical_crossentropy", optimizer="adam",  metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32)

# test set score
score = model.evaluate(x_test, y_test)
print('test set metrics: ', score)

### Predictions
text = "Mary said she had been affected much the same way. Sid seemed satisfied.Tom got out of the presence as quick as he plausibly could, and after that he complained of toothache for a week, and tied up his jaws everynight. He never knew that Sid lay nightly watching, and frequently slipped the bandage free and then leaned on his elbow listening a good while at a time, and afterward slipped the bandage back to its place again. Tom's distress of mind wore off gradually and the toothache grew irksome and was discarded. If Sid really managed to make anything out of Tom's disjointed mutterings, he kept it to himself."

# separating all the characters
prediction_corpus = list(text)

# creating sets of length seq_len 
corpus = [prediction_corpus[i:i+seq_len] for i in range(0, len(prediction_corpus), seq_len)]

# making lengths of sets equal by appending space
for i in range(len(corpus)):
    if len(corpus[i]) != seq_len:
        corpus[i].append(' ')

# loop over all sets
for l in range(len(corpus)):
    # loop over seq_len
    for i in range(seq_len):
        # sample will store the index of all the letters in a set
        sample = [chr_to_id[c] for c in corpus[l]]
        # convert and reshape to ndarray
        sample = np.array(sample).reshape(1, seq_len, 1)
        # normalization
        sample = sample/float(len(vocab))
        # prediction
        p = np.argmax(model.predict(sample))
        # add predicted word to set
        corpus[l].append(id_to_chr[p])
        # shift the letters to the left by removing the first letter
        del(corpus[l][0])
        

    print(corpus[l])