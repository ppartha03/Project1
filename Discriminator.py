from Encoder_Decoder import *

#discriminator class

class Discriminator(Encoder):
    def __init__(self):
        #self.Encoder = Encoder_Decoder.Encoder()
        Encoder.__init__(self)
        self.V_d = theano.shared(self.sample_weights(self.vocab_size, self.embedding_dim))
        self.b_V_d = theano.shared(np.ones(self.embedding_dim, dtype=dtype))
        self.params = self.eparams + [self.V_d, self.b_V_d]
        self.gparams = []
        self.train = []
        self.test = []

#    def load_data(self, filename ):
#        [self.train, self.test] = pickle.load(open(filename,'r'))

    def load_weights(self, filename):
        self.params = pickle.load(open(filename,'r')

    def sample(self,y_t, type = 'max'):
        if type == 'max':
            return y_t.index(max(y_t))
        elif type == 'sample':
            return np.random.multinomial(1,y_t,size=1).argmax()

    def run_discriminator(self, utterances):
        # initialize weights
        # i_t and o_t should be "open" or "closed"
        # f_t should be "open" (don't forget at the beginning of training)
        # we try to archive this by appropriate initialization of the corresponding biases



        [h_vals, _], _ = theano.scan(fn=self.Encoder.sentence_encoder,
                                  sequences = self.v,
                                  outputs_info = [self.h0, self.c0 ], # corresponds to return type of fn
                                  non_sequences = [None] )

        [h1_vals,_], _ = theano.scan(fn=self.Encoder.utterance_Encoder,
                                  sequences =h_vals,
                                  outputs_info = [h0, c0 ], # corresponds to return type of fn
                                  non_sequences = [None] )

        p = sigma(theano.dot(h1_vals,self.V_d) + self.b_V_d)
        self.cost = -T.mean(self.target * T.log(p[0])+ (1.- self.target) * T.log(1. - p[1]))

        self.lr = np.cast[dtype](self.lr)
        learning_rate = theano.shared(self.lr)


        for param in self.params:
          gparam = T.grad(self.cost, param)
          self.gparams.append(gparam)

        self.updates=[]
        for param, gparam in zip(self.params, self.gparams):
            updates.append((param, param - gparam * learning_rate))

        self.sentence_encoding = theano.function([self.v],h_vals)

        h_vec =[]
        for v in utterances:
            h_vec + = [self.sentence_encoding(v)]
        h_vectors = theano.tensor.stack(h_vec)
        self.get_cost = theano.function([h_vals,self.target], self.cost,updates = updates)
        return self.get_cost([h_vectors,target])
