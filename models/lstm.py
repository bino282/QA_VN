import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot,Conv2D,Flatten,MaxPool2D
from keras.optimizers import Adam
from layers import Match,MatchTensor
from layers.SpatialGRU import *

class LSTM_MATCH():
    def __init__(self,config):
        self.config = config
        self.model = self.build()
    
    def build(self):
        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'],weights=[self.config['embed']], trainable = self.config['embed_trainable'])

        seq1_embed = embedding(seq1)
        seq1_embed = Dropout(rate = self.config['dropout_rate'])(seq1_embed)
        seq2_embed = embedding(seq2)
        seq2_embed = Dropout(rate = self.config['dropout_rate'])(seq2_embed)

        lstm1 = Bidirectional(LSTM(self.config['hidden_size']))
        lstm2 = Bidirectional(LSTM(self.config['hidden_size']))
        seq1_rep = lstm1(seq1_embed)
        seq2_rep = lstm2(seq2_embed)

        final_rep = concatenate([seq1_rep,seq2_rep])

        out = Dense(2, activation='softmax')(final_rep)

        model = Model(inputs=[seq1, seq2], outputs=out)
        return model