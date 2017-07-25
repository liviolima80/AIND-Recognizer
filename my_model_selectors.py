import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self, alpha=1.0):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    p = probabilities of state transition = state_n * (state_n - 1) +
        probabilites of initial distribution = (state_n - 1) +
        mean of features distribution = state_n * len(self.X[0]) +
        standard deviation of df features distribution = state_n * len(self.X[0])
    """

    def select(self, alpha=1.0):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_model = None
        best_score = 1e10
        X, lengths = self.hwords[self.this_word]

        for state_n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=state_n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
                logL = model.score(X, lengths)
                N = len(self.X)
                p = state_n * state_n + 2 * state_n * len(self.X[0]) - 1
                score = -2 * logL  + alpha * p * np.log(N)
                if (score < best_score):
                    best_score = score
                    best_model = model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self, alpha=1.0):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_model = None
        best_score = -1e10

        X, lengths = self.hwords[self.this_word]
        M = len(self.words.keys())

        for state_n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=state_n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
                logL = model.score(X, lengths)

                antiL = 0.0
                for m in self.words.keys():
                    if m is not self.this_word:
                        X_anti, lengths_anti = self.hwords[m]
                        antiL += model.score(X_anti, lengths_anti)

                score = logL - alpha*float(1/float(M-1)) * antiL
                if (score > best_score):
                    best_score = score
                    best_model = model
            except:
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self, alpha=1.0):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_model = 2
        best_score = -1e10

        X, lengths = self.hwords[self.this_word]

        if(len(self.sequences) > 1):

            for state_n in range(self.min_n_components, self.max_n_components+1):
                split_method = KFold(min(len(self.sequences), 3))
                folds = split_method.split(self.sequences)
                try:
                    sum_logL_test = 0
                    for cv_train_idx, cv_test_idx in folds:
                        # train
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        model = GaussianHMM(n_components=state_n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train, lengths_train)

                        # test
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        logL_test = model.score(X_test, lengths_test)
                        sum_logL_test += logL_test

                    if (sum_logL_test > best_score):
                        best_score = sum_logL_test
                        best_model = state_n
                except:
                    pass

            X, lengths = self.hwords[self.this_word]
            return GaussianHMM(n_components=best_model, covariance_type="diag", n_iter=1000,
                               random_state=self.random_state, verbose=False).fit(X, lengths)

        else:
            X, lengths = self.hwords[self.this_word]
            return GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
