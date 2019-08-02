from datetime import datetime
import traceback
import keras.backend as K
import tensorflow as tf
from keras import models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model

import os,sys

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)
from utils import *
import argparse
from models import lstm_cnn,lstm,mvrnn

def main(args):


    # Training params:
    max_len = args.max_len
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr

    path = "./data/train.txt"
    path_validation = "./data/dev.txt"
    path_test = "./data/test.txt"

    # Create dataframes
    print ("\nReading training data:")
    X1_train,X2_train,Y_train = read_data_from_file(path)
    print("num_pairs_train : ",len(X1_train))
    print("sample data : ")
    print(X2_train[0:10])
    print ("\nReading validation data: ")
    X1_dev,X2_dev,Y_dev = read_data_from_file(path_validation)
    print("num_pairs_dev : ",len(X1_dev))
    print ("\nReading test data: ")
    X1_test,X2_test,Y_test = read_data_from_file(path_test)
    print("num_pairs_test : ",len(X1_test))
    vocab,voc2index = creat_voc(X1_train+X2_train,min_count = 5)
    print("vocab_len : ",len(voc2index))

    # load embed matrix
    embed_matrix = create_embedd("word_vector.bin",vocab)
    print(embed_matrix.shape)

    # Convert data to index and padding
    X1_train_pad = convert_and_pad(X1_train,voc2index,max_len)
    X2_train_pad = convert_and_pad(X2_train,voc2index,max_len)
    X1_dev_pad = convert_and_pad(X1_dev,voc2index,max_len)
    X2_dev_pad = convert_and_pad(X2_dev,voc2index,max_len)
    X1_test_pad = convert_and_pad(X1_test,voc2index,max_len)
    X2_test_pad = convert_and_pad(X2_test,voc2index,max_len)


    # Optimization algorithm used to update network weights
    optimizer = Adam(lr=learning_rate, epsilon=1e-8, clipnorm=2.0)


    model_config={'seq1_maxlen':max_len,'seq2_maxlen':max_len,
                'vocab_size':len(voc2index),'embed_size':300,
                'hidden_size':300,'dropout_rate':0.2,
                'embed':embed_matrix,
                'embed_trainable':True}
    try:
        model_matching = load_model("./model_saved/model-lstm-cnn.h5")
        print("Load model success......")
    except:
        print("Creating new model......")
        model_matching = lstm_cnn.MATCH_LSTM_CNN(config=model_config).model
    print(model_matching.summary())

    model_matching.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    checkpoint = ModelCheckpoint("./model_save/model-lstm-cnn-{epoch:02d}-{val_acc:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=3)

    MAP_last = 0

    for epoch in range(150):
        print('Train on iteration {}'.format(epoch))
        model_matching.fit([X1_train_pad,X2_train_pad],Y_train,batch_size=batch_size,epochs=1,
                    validation_data=([X1_dev_pad,X2_dev_pad],Y_dev))
        y_dev_pred = model_matching.predict([X1_dev_pad,X2_dev_pad])
        MAP_dev,MRR_dev = map_score(X1_dev,X2_dev,y_dev_pred,Y_dev)
        print('MAP_dev = {}, MRR_dev = {}'.format(MAP_dev,MRR_dev))
        if(MAP_dev>MAP_last):
            model_matching.save('./model_save/model-lstm-cnn.h5')
            print('Model saved !')
            MAP_last = MAP_dev                                              
        y_test_pred = model_matching.predict([X1_test_pad,X2_test_pad])
        MAP_test,MRR_test = map_score(X1_test,X2_test,y_test_pred,Y_test)
        print('MAP_test = {}, MRR_test = {}'.format(MAP_test,MRR_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training params:
    parser.add_argument('--max_len', type=int, default=50,
                        help='Number of files in one batch.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of files in one batch.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    args = parser.parse_args()

    main(args)