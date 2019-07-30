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
from models import mvrnn

def main(args):


    # Training params:
    max_len = args.max_len
    batch_size = args.batch_size
    input_epoch_length = args.epoch_len
    epochs = args.epochs
    learning_rate = args.lr
    log_file = args.log_file

    path = "./data/train.txt"
    path_validation = "./data/dev.txt"
    path_test = "./data/test.txt"

    # Create dataframes
    print ("\nReading training data:")
    X1_train,X2_train,Y_train = read_data_from_file(path)
    print("num_pairs_train : ",len(X1_train))
    print ("\nReading validation data: ")
    X1_dev,X2_dev,Y_dev = read_data_from_file(path_validation)
    print("num_pairs_dev : ",len(X1_dev))
    print ("\nReading test data: ")
    X1_test,X2_test,Y_test = read_data_from_file(path_test)
    print("num_pairs_test : ",len(X1_test))
    vocab,voc2index = creat_voc(X1_train+X2_train,min_count = 5)
    print("vocab_len : ",len(voc2index))

    # Convert data to index and padding
    X1_train_pad = convert_and_pad(X1_train,voc2index,max_len)
    X2_train_pad = convert_and_pad(X2_train,voc2index,max_len)
    X1_dev_pad = convert_and_pad(X1_dev,voc2index,max_len)
    X2_dev_pad = convert_and_pad(X2_dev,voc2index,max_len)
    X1_test_pad = convert_and_pad(X1_test,voc2index,max_len)
    X2_test_pad = convert_and_pad(X2_test,voc2index,max_len)


    # Optimization algorithm used to update network weights
    optimizer = Adam(lr=learning_rate, epsilon=1e-8, clipnorm=2.0)

    # Loss-function for compiling model
    loss = {'ctc': lambda y_true, y_pred: y_pred}


    model_config={'seq1_maxlen':max_len,'seq2_maxlen':max_len,
                'vocab_size':len(voc2index),'embed_size':300,
                'hidden_size':300,'dropout_rate':0.5,
                'embed_trainable':True}
    try:
        model_matching = load_model("./model_saved/model-lstm-cnn.h5",custom_objects={'Position_Embedding':Position_Embedding,'Attention':Attention})
        print("Load model success......")
    except:
        print("Creating new model......")
        model_matching = mvrnn.MVRNN(config=model_config).model
    print(model_matching.summary())
    optimize = Adam(lr=0.0001)
    model_matching.compile(loss='sparse_categorical_crossentropy',optimizer=optimize,metrics=['accuracy'])
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
            model_matching.save('./model_saved/model-lstm-cnn.h5')
            print('Model saved !')
            MAP_last = MAP_dev                                              
        y_test_pred = model_matching.predict([X1_test_pad,X2_test_pad])
        MAP_test,MRR_test = map_score(X1_test,X2_test,y_test_pred,Y_test)
        print('MAP_test = {}, MRR_test = {}'.format(MAP_test,MRR_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training params:
    parser.add_argument('--max_len', type=int, default=100,
                        help='Number of files in one batch.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of files in one batch.')
    parser.add_argument('--epoch_len', type=int, default=32,
                        help='Number of batches per epoch. 0 trains on full dataset.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--log_file', type=str, default="log",
                        help='Path to log stats to .csv file.')

    args = parser.parse_args()

    main(args)