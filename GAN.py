from Generator import *
from Discriminator import *

def generate_noisy_dataset():
    Gen = Generator()
    Gen.load_params(generator_weights)
    contexts = pickle.load(open("contexts",'r'))

    for c in contexts.iterkeys():
        contexts[c] = Gen.res_gen([c])
    return contexts

def pre_train_generator(datafile)
    #dataset will be a dictionary of context and response pairs
    #This will be used to pretrain the generator on Log-Likelihood
    dataset = pickle.load(open(datafile,'r'))
    Gen = Generator(cost='tf')
    prev_cost = 100.

    while prev_cost>epsilon:
        total_cost =0.
        for c,r in dataset.iteritems():
            total_cost + = Gen.tf_train_cost([c,r])
        print "cost after Epoch:" + str(e) + "is "+str(total_cost/size)
        prev_cost = total_cost
    Gen.save_params('nll')

def pre_train_discriminator(datafile):
    Dis = Discriminator()
    dataset = pickle.load(open(datafile,'r'))
    prev_cost = 100.

    while prev_cost>epsilon:
        total_cost =0.
        for c,r in dataset.iteritems():
            total_cost + = Dis.run_discriminator([c,r])
        print "Discriminator cost after Epoch:" + str(e) + "is "+str(total_cost/size)
        prev_cost = total_cost

def adversarial_training():
    
