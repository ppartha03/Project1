from Encoder_Decoder import *
from Discriminator import *

#discriminator class

class Generator(Encoder,Decoder):
    def __init__(self,vocab_file,cost = 'gan'):
        Encoder.__init__(self)
        Decoder.__init__(self)
        self.vocab = pickle.load(vocab_file)
        self.cost = cost
        self.evaluator = Discriminator()
        self.params = self.eparams + self.dparams
        self.gparams = []
        self.train = []
        self.test = []

    def load_data(self, filename ):
        [self.train, self.test] = pickle.load()

    def init_params(self,filename):
        self.params = pickle.load(filename)

    def sample(self,y_t, type = 'max'):
        if type == 'max':
            return y_t.index(max(y_t))
        elif type == 'sample':
            return np.random.multinomial(1,y_t,size=1).argmax()

    def MC(self,context, word_embed, response):
        #MC Rollout using Sampling -not Beam search-
        context_mc = context
        word_embed_mc = word_embed
        response_encoding = response
        while token!='<eor>':
            #save context embedding and use the same to be passed to discriminator
            [word_embed_mc,context_mc,y_t] = self.Decoder.sentence_decoder(word_embed_mc, context_mc, _)
            word_token = self.sample(y_t,type='max')
            response_encoding += [word_token]
        return response_encoding

    def save_params(self,text):
        pickle.dump(self.params,open("Generator_params_"+text))
    def run_generator(self,):
        # initialize weights
        # i_t and o_t should be "open" or "closed"
        # f_t should be "open" (don't forget at the beginning of training)
        # we try to archive this by appropriate initialization of the corresponding biases



        [h_vals, _], _ = theano.scan(fn=self.Encoder.sentence_encoder,
                                  sequences = self.x_t,
                                  outputs_info = [self.h0, self.c0 ], # corresponds to return type of fn
                                  non_sequences = [None] )

        [h1_vals,_], _ = theano.scan(fn=self.Encoder.utterance_Encoder,
                                  sequences =h_vals,
                                  outputs_info = [self.h0, self.c0 ], # corresponds to return type of fn
                                  non_sequences = [None] )

        #p = sigma(theano.dot(h1_vals,self.V_d) + self.b_V_d)
        #    cost = -T.mean(target * T.log(p[0])+ (1.- target) * T.log(1. - p[1]))
        self.response = []
        token = ''
        b=0.
        count =0
        self.cost=0.
        self.disc_loss = 0.
        while token!='<eor>':
            #save context embedding and use the same to be passed to discriminator
            [self.x_t,self.h0,y_t] = self.Decoder.sentence_decoder(stack(self.x_t,h1_vals), self.h0, _)
            word_token = self.sample(y_t,type='max')
            self.response+=[word_token]
        #    if self.cost = 'tf':
        #Log-Likelihood
            self.cost + = - (count*cost + T.mean(self.target[count]*T.log(y_t) + (1.-self.target[count])*T.log(1.-y_t)))/(count+1.)
        #elif self.cost = 'gan':
        #GAN-training -REinforce
            RO_response = self.MC_rollout(context = self.h0,word_embed = self.x_t,self.response)
            [response_embed,_] = theano.scan(fn=self.Encoder.sentence_encoder,
                                        sequences = RO_response,
                                        outputs_info = [self.h0,self.c0]
                                        non_sequences = None)
            utterance_embed = theano.tensor.stack(response_embed,self.context)
            cost_d = self.evaluator.run_discriminator(utterances)
            self.disc_loss += -(count*disc_loss + T.mean(T.log(y_t)*(cost_d - b)))/(count + 1.)
            b = (count*b + cost_d)/(count + 1.)
            count += 1
            self.lr = np.cast[dtype](self.lr)
            learning_rate = theano.shared(self.lr)

        self.res_gen = theano.function([self.x_t],self.response,updates = None)

    def get_updates(self,flag='gan'):

        if flag == "gan":
            for param in self.params:
              gparam = T.grad(self.disc_loss, param)
              self.gparams.append(gparam)

              self.updates=[]
            for param, gparam in zip(self.params, self.gparams):
                self.updates.append((param, param - gparam * learning_rate))

        elif flag == "tf":
            for param in self.params:
              gparam = T.grad(cost, param)
              self.gparams.append(gparam)

            self.updates=[]
            for param, gparam in zip(self.params, self.gparams):
                self.updates.append((param, param - gparam * learning_rate))

    self.gan_train_cost = theano.function([self.x_t,self.h0],self.disc_loss,updates = self.get_updates(self.cost))

    self.tf_train_cost = theano.function([self.x_t,self.h0],self.cost,updates = self.get_updates(self.cost))
