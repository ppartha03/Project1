# Discriminator #
# inputs: context c_i (text), r_i/r* (text)
# outputs: 0/1 (probability if the response is sampled from the true distribution or model distribution)

import theano
import numpy
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def sample_weights(sizeX, sizeY):
    values = np.ndarray([sizeX, sizeY], dtype=dtype)
    for dx in xrange(sizeX):
        vals = np.random.uniform(low=-1./sizeX, high=1./sizeX,  size=(sizeY,))
        #vals_norm = np.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value
    values = values / svs[0]
    return values

class Encoder:
    sigma = lambda x: 1 / (1 + T.exp(-x))

    def __init__(self, vocab_size = 1000,lr = 0.1, n_in = 7, n_y=7, n_yh=1, n_hidden =10, n_i =10, n_c =10, n_o = 10 ,n_f = 10):
        self.n_in = n_in
        self.n_y = n_y
        self.n_yh = n_yh
        self.n_hidden = n_hidden
        self.n_i = n_i
        self.n_c = n_c
        self.n_o = n_o
        self.n_f = n_f
        self.lr = lr
        self.vocab_size = vocab_size
        self.embedding_dim = int(vocab_size/2)
        #self.type = 'dis'

        self.V = theano.shared(self.sample_weights(self.vocab_size, self.embedding_dim))
        self.b_V = theano.shared(np.ones(self.embedding_dim, dtype=dtype))
        self.W_xi = theano.shared(self.sample_weights(n_in, n_i))
        self.W_hi = theano.shared(self.sample_weights(n_hidden, n_i))
        self.W_ci = theano.shared(self.sample_weights(n_c, n_i))
        self.b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_i)))
        self.W_xf = theano.shared(self.sample_weights(n_in, n_f))
        self.W_hf = theano.shared(self.sample_weights(n_hidden, n_f))
        self.W_cf = theano.shared(self.sample_weights(n_c, n_f))
        self.b_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_f)))
        self.W_xc = theano.shared(self.sample_weights(n_in, n_c))
        self.W_hc = theano.shared(self.sample_weights(n_hidden, n_c))
        self.b_c = theano.shared(np.zeros(n_c, dtype=dtype))
        self.W_xo = theano.shared(self.sample_weights(n_in, n_o))
        self.W_ho = theano.shared(self.sample_weights(n_hidden, n_o))
        self.W_co = theano.shared(self.sample_weights(n_c, n_o))
        self.b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_o)))
        self.W_hy = theano.shared(self.sample_weights(n_hidden, n_y))
        self.b_y = theano.shared(np.zeros(n_y, dtype=dtype))

        self.W2_xi = theano.shared(self.sample_weights(n_in, n_i))
        self.W2_hi = theano.shared(self.sample_weights(n_hidden, n_i))
        self.W2_ci = theano.shared(self.sample_weights(n_c, n_i))
        self.b2_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_i)))
        self.W2_xf = theano.shared(self.sample_weights(n_in, n_f))
        self.W2_hf = theano.shared(self.sample_weights(n_hidden, n_f))
        self.W2_cf = theano.shared(self.sample_weights(n_c, n_f))
        self.b2_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_f)))
        self.W2_xc = theano.shared(self.sample_weights(n_in, n_c))
        self.W2_hc = theano.shared(self.sample_weights(n_hidden, n_c))
        self.b2_c = theano.shared(np.zeros(n_c, dtype=dtype))
        self.W2_xo = theano.shared(self.sample_weights(n_in, n_o))
        self.W2_ho = theano.shared(self.sample_weights(n_hidden, n_o))
        self.W2_co = theano.shared(self.sample_weights(n_c, n_o))
        self.b2_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_o)))
        self.W2_hy = theano.shared(self.sample_weights(n_hidden, n_yh))
        self.b2_y = theano.shared(np.zeros(n_yh, dtype=dtype))

        self.c0 = theano.shared(np.zeros(n_hidden, dtype=dtype))
        self.h0 = T.tanh(c0)
        self.eparams = [self.V, self.b_V, self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.W_hy, self.b_y, self.c0, self.h_t, self.h2_tm1, self.c2_tm1, self.W2_xi, self.W2_hi, self.W2_ci, self.b2_i, self.W2_xf, self.W2_hf, self.W2_cf, self.b2_f, self.W2_xc, self.W2_hc, self.b2_c, self.W2_xy, self.W2_ho, self.W2_cy, self.b2_o, self.W2_hy, self.b2_y]
        self.gparams = []

    def load_data(self,filename):
        #self.train =
        #self.test =

# for the other activation function we use the tanh
    act = T.tanh
    def sentence_encoder(self,x_t, h_tm1, c_tm1):
        x_t = sigma(theano.dot(x_t,self.V) + self.b_V)
        i_t = sigma(theano.dot(x_t, self.W_xi) + theano.dot(self.h_tm1, self.W_hi) + theano.dot(self.c_tm1, self.W_ci) + self.b_i)
        f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(self.h_tm1, self.W_hf) + theano.dot(self.c_tm1, self.W_cf) + self.b_f)
        c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, self.W_xc) + theano.dot(h_tm1, self.W_hc) + self.b_c)
        o_t = sigma(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho) + theano.dot(c_t, self.W_co)  + self.b_o)
        h_t = o_t * act(c_t)
        #y_t = sigma(theano.dot(h_t, self.W_hy) + self.b_y)

        return [h_t, c_t]

    def utterance_Encoder(self, h_t,h2_tm1, c2_tm1):
        i2_t = sigma(theano.dot(h_t, self.W2_xi) + theano.dot(h2_tm1, self.W2_hi) + theano.dot(c2_tm1, self.W2_ci) + self.b2_i)
        f2_t = sigma(theano.dot(h_t, self.W2_xf) + theano.dot(h2_tm1, self.W2_hf) + theano.dot(c2_tm1, self.W2_cf) + self.b2_f)
        c2_t = f2_t * c2_tm1 + i2_t * act(theano.dot(h_t, self.W2_xc) + theano.dot(h2_tm1, self.W2_hc) + self.b2_c)
        o2_t = sigma(theano.dot(h_t, self.W2_xo)+ theano.dot(h2_tm1, self.W2_ho) + theano.dot(c2_t, self.W2_co)  + self.b2_o)
        h2_t = o2_t * act(c2_t)
        #y2_t = sigma(theano.dot(h2_t, self.W2_hy) + self.b2_y)
    return [ h2_t, c2_t]


    #    cost = -T.mean(target * T.log(y_vals)+ (1.- target) * T.log(1. - y_vals))

    #    self.lr = np.cast[dtype](self.lr)
    #    learning_rate = theano.shared(self.lr)


    #    for param in self.params:
    #      gparam = T.grad(cost, param)
    #      self.gparams.append(gparam)

    #    updates=[]
    #    for param, gparam in zip(params, gparams):
    #        updates.append((param, param - gparam * learning_rate))


    #    learn_lstm = theano.function( input = [v,target], output =cost, updates = updates)


class Decoder:

    def __init__(self, lr = 0.1, n_in = 14, n_y=7, n_hidden =20, n_i =10, n_c =10, n_o = 10 ,n_f = 10):
        self.n_in = n_in
        self.n_y = n_y
        self.n_hidden = n_hidden
        self.n_i = n_i
        self.n_c = n_c
        self.n_o = n_o
        self.n_f = n_f
        self.lr = lr
        #self.type = type_

        self.W_xi = theano.shared(self.sample_weights(n_in, n_i))
        self.W_hi = theano.shared(self.sample_weights(n_hidden, n_i))
        self.W_ci = theano.shared(self.sample_weights(n_c, n_i))
        self.b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_i)))
        self.W_xf = theano.shared(self.sample_weights(n_in, n_f))
        self.W_hf = theano.shared(self.sample_weights(n_hidden, n_f))
        self.W_cf = theano.shared(self.sample_weights(n_c, n_f))
        self.b_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_f)))
        self.W_xc = theano.shared(self.sample_weights(n_in, n_c))
        self.W_hc = theano.shared(self.sample_weights(n_hidden, n_c))
        self.b_c = theano.shared(np.zeros(n_c, dtype=dtype))
        self.W_xo = theano.shared(self.sample_weights(n_in, n_o))
        self.W_ho = theano.shared(self.sample_weights(n_hidden, n_o))
        self.W_co = theano.shared(self.sample_weights(n_c, n_o))
        self.b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_o)))
        self.W_hy = theano.shared(self.sample_weights(n_hidden, n_y))
        self.b_y = theano.shared(np.zeros(n_y, dtype=dtype))
        self.context = theano.shared(np.zeros(n_y, dtype=dtype))
        self.x_t = theano.shared(np.zeros(n_y, dtype=dtype))
        self.dparams = [self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.W_hy, self.b_y, self.c0, self.h_t]
        self.gparams = []

        self.c0 = theano.shared(np.zeros(n_hidden, dtype=dtype))
        self.h0 = T.tanh(c0)

        def  initialize_context(self,context):
            self.context = context


        def sentence_decoder(self,x_t, h_tm1, c_tm1):
            x_t = T.stack(self.context, x_t)
            i_t = sigma(theano.dot(x_t, self.W_xi) + theano.dot(self.h_tm1, self.W_hi) + theano.dot(self.c_tm1, self.W_ci) + self.b_i)
            f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(self.h_tm1, self.W_hf) + theano.dot(self.c_tm1, self.W_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, self.W_xc) + theano.dot(h_tm1, self.W_hc) + self.b_c)
            o_t = sigma(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, W_ho) + theano.dot(c_t, self.W_co)  + self.b_o)
            h_t = o_t * act(c_t)
            y_t = sigma(theano.dot(h_t, self.W_hy) + self.b_y)

            return [h_t, c_t,y_t]
