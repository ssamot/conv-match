s__author__ = 'ssamot'

from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, TimeDistributedDense, Dense, Merge, Dropout
from keras.models import Sequential
from keras.layers import Merge


from layers import AttentionMerge, AttentionRecurrent, AttentionPooling, ResettingRNN
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.constraints import MaxNorm
from keras.layers.noise import GaussianNoise
from keras.regularizers import WeightRegularizer
from keras.layers.convolutional import  MaxPooling1D, AveragePooling1D, Convolution1D
from keras import activations
import theano.tensor as T
from utils import bcolors
from keras.utils.visualize_util import plot
from keras.layers import Reshape, Permute, Layer, InputSpec, InputLayer
from keras import backend as K
import numpy as np


size = 2**7
attention = size
alpha = 1.0
learning_rate = 0.001
convolution = True


init_function = "glorot_normal"

def elu(x):
    pos = K.relu(x)
    neg = (x - abs(x)) * 0.5
    x =  pos + 1.0 * (K.exp(neg) - 1.)
    #
    return x

def leakyrelu(x):
    x =  K.relu(x, alpha=0.3, max_value = 2)
    return T.minimum(x, 2)

activations.elu = elu
activations.leakyrelu = leakyrelu

act = "elu"

def reg():
    return None
    #return WeightRegularizer(l2 = 0.002, l1 = 0)

def mx():
    return None
    #return MaxNorm(3.0)

# def bn():
#     #return BatchNormalization(mode = 0)
#     return BatchNormalization(mode = 1, momentum=0.9)



class Logic():
    def __init__(self, embed_hidden_size=size, sent_hidden_size=size, query_hidden_size=size,
                 deep_hidden_size=size, RNN=SimpleRNN):
        self.deep_hidden_size = deep_hidden_size

        self.embed_hidden_size = embed_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.query_hidden_size = query_hidden_size
        self.RNN = RNN




    def _getopt(self):
        from keras.optimizers import adam, rmsprop, sgd, adadelta, nadam
        opt = adam(lr=learning_rate)
        #opt = rmsprop(learning_rate)
        #opt = adadelta()
        #opt = sgd(learning_rate = 0.01, momentum= 0.8, nesterov=True)
        return opt



    def distancenet(self, vocab_size, output_size,  maxsize = 1, hop_depth = -1, dropout = False, d_perc = 1,  type = "CCE", shape = 0, q_shape = 1):
        print(bcolors.UNDERLINE + 'Building nn model...' + bcolors.ENDC)

        print(q_shape, shape, "====================")
        sentrnn = Sequential()
        emb = Embedding(vocab_size, self.embed_hidden_size, mask_zero=False,W_constraint=mx(), W_regularizer=reg(), init = init_function, input_shape=shape)
        sentrnn.add(emb)
        #emb_bn = bn()
        #sentrnn.add(emb_bn)
        sentrnn.add(Dropout(0.2))

        if(convolution):
            #print maxsize, q_shape
            #exit()
            conv = Convolution1D(self.sent_hidden_size,maxsize, subsample_length = 1, activation=act,border_mode="same")
            #conv_bn = bn()
            sentrnn.add(conv)
            #sentrnn.add(conv_bn)
            #sentrnn.add(Dropout(0.1))
            sentrnn.add(MaxPooling1D(pool_length=maxsize))
            #sentrnn.add(Dropout(d_perc))
        else:
            sentrnn.add(MaxPooling1D(pool_length=maxsize))
            #sentrnn.add(SimpleRNN( self.query_hidden_size, return_sequences=True,activation = "leakyrelu", init = init_function,dropout_W = 0.3, dropout_U=0.3, consume_less = "mem"))
            # sentrnn.add(ResettingRNN(maxsize, self.query_hidden_size, return_sequences=True,activation = "tanh", init = init_function,dropout_W = 0.3, dropout_U=0.3, consume_less = "mem"))
            # sentrnn.add(bn())
            # sentrnn.add(AttentionPooling(pool_length=maxsize))

            #sentrnn.add(Convolution1D(self.sent_hidden_size,maxsize, subsample_length = 1, activation=act, border_mode="same"))
            #sentrnn.add(bn())
            #
            #sentrnn.add(Dropout(0.1))
            #td = TimeDistributedDense(self.sent_hidden_size, activation = act, init = init_function)
            #sentrnn.add(td)
            #sentrnn.add(bn())
            #sentrnn.add(Dropout(d_perc))


        qrnn = Sequential()
        #emb = Embedding(vocab_size, self.embed_hidden_size, mask_zero=False,W_constraint=mx(), W_regularizer=reg(), init = init_function, input_shape=shape, input_length=q_shape)

        qrnn.add(InputLayer(input_shape=(q_shape,)))
        qrnn.add(emb)
        #qrnn.add(emb_bn)
        qrnn.add(Dropout(0.2))



        if(convolution):
            #conv = Convolution1D(self.sent_hidden_size,q_shape, subsample_length = 1, activation=act, border_mode="same")
            qrnn.add(conv)
            #qrnn.add(conv_bn)
            #qrnn.add(Dropout(0.1))
            qrnn.add(MaxPooling1D(pool_length=q_shape))
            qrnn.add(Flatten(1))
        else:
            qrnn.add(SimpleRNN( self.query_hidden_size, return_sequences=False,activation = act, init = init_function,dropout_W = 0.1, dropout_U=0.1, consume_less = "mem"))
            #qrnn.add(Convolution1D(self.sent_hidden_size,q_shape, subsample_length = 1, activation=act, border_mode="same"))
            #qrnn.add(bn())
            #qrnn.add(MaxPooling1D(pool_length=q_shape))


            #qrnn.add(Flatten())




        init_qa = [sentrnn, qrnn]
        past = []



        #at = GRU(self.sent_hidden_size, dropout_W = 0.3, dropout_U=0.3, activation=act)

        td = TimeDistributedDense(self.sent_hidden_size, activation = act, init = init_function)
        at = AttentionRecurrent(self.sent_hidden_size, activation= "softmax")



        for i in range(hop_depth):
            hop = Sequential()
            hop.add(AttentionMerge((i == hop_depth-1), "abs", init_qa + past, mode = "sum"))
            hop.add(td)
            #hop.add(bn())
            #hop.add(Dropout(0.1))
            hop.add(at)

            #hop.add(bn())
            #hop.add(Dropout(0.1))
            past.append(hop)

        # model = Sequential()
        # model.add(AttentionMerge(False, "concat", init_qa + past, mode = "concat"))
        # model.add(LSTM(self.query_hidden_size, return_sequences=False, init = init_function,dropout_W = 0.2, dropout_U=0.2))



        model = hop



        #self._adddepth(model, dropout, d_perc)
        model.add(Dense( vocab_size,W_constraint=mx(), W_regularizer=reg(), init = init_function))
        #model.add(bn())
        model.add(Activation("softmax"))
        if(type == "CCE"):
            model.compile(optimizer=self._getopt(), loss='categorical_crossentropy', metrics=["accuracy"], class_mode='categorical')
        else:
            model.compile(optimizer=self._getopt(), loss='mse')


        print model.summary()
        plot(model, show_layer_names=False, show_shapes=True, to_file='model.png')

        return model






    def nomemory(self, vocab_size, output_size, dropout = True, d_perc = 1,  type = "CCE"):
        print(bcolors.UNDERLINE + 'Building nn model...' + bcolors.ENDC)

        sentrnn = Sequential()
        sentrnn.add(Embedding(vocab_size, self.embed_hidden_size, mask_zero=True))
        sentrnn.add(self.RNN(self.sent_hidden_size,return_sequences=False, activation = "relu"))




        qrnn = Sequential()
        qrnn.add(Embedding(vocab_size, self.embed_hidden_size, mask_zero=True))
        qrnn.add(self.RNN( self.query_hidden_size, return_sequences=False, activation = "relu"))


        model = Sequential()
        model.add(Merge([sentrnn, qrnn], mode='concat'))




        model.add(Dense(self.deep_hidden_size))
        model.add(LeakyReLU())



        #self._adddepth(model, output_size, dropout, d_perc, softmax = (type == "CCE"))

        model.add(Dense( vocab_size,W_constraint=mx(), W_regularizer=reg(), init = init_function, activation="softmax"))

        if(type == "CCE"):
            model.compile(optimizer=self._getopt(), loss='categorical_crossentropy', class_mode='categorical')
        else:
            model.compile(optimizer=self._getopt(), loss='mse')


        return model












    def __repr__(self):
        metadata = ('RNN / Embed / Sent / Query / Hidden = {}, {}, {}, {}, {}'.format(self.RNN, self.embed_hidden_size,
                                                                                      self.sent_hidden_size,
                                                                                      self.query_hidden_size,
                                                                                      self.deep_hidden_size))
        return metadata


class Flatten(Layer):
    '''Flattens the input. Does not affect the batch size.

    # Example

    ```python
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    '''
    def __init__(self, l, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        super(Flatten, self).__init__(**kwargs)
        self.l = l

    def get_output_shape_for(self, input_shape):
        input_shape = list(input_shape[:])
        input_shape[1] = self.l
        return (input_shape[0], np.prod(input_shape[1:]))

    def call(self, x, mask=None):
        return K.batch_flatten(x)