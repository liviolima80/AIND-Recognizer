import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for i in test_set._hmm_data:
        X, lenght = test_set._hmm_data[i]
        #print(i, lenght)
        prob = {}
        for w in models:
            try:
                logL = models[w].score(X, lenght)
                #print(logL)
                prob[w] = logL
            except:
                #print('score error')
                prob[w] = -float('Inf')
        max_w = max(prob, key=prob.get)
        #print(max_w)
        probabilities.append(prob)
        guesses.append(max_w)
    return probabilities, guesses

""""
def recognize_lm(models: dict, test_set: SinglesData, lm, lm_factor):

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for i in test_set._hmm_data:
        X, lenght = test_set._hmm_data[i]
        #print(i, lenght)
        prob = {}
        for w in models:
            try:
                logL = models[w].score(X, lenght) + float(lm_factor) * lm.log_p(w)
                #print(logL)
                prob[w] = logL
            except:
                #print('score error')
                prob[w] = -float('Inf')
        max_w = max(prob, key=prob.get)
        #print(max_w)
        probabilities.append(prob)
        guesses.append(max_w)
    return probabilities, guesses
"""