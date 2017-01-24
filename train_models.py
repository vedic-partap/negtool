import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from statsmodels.sandbox.stats.runs import mcnemar
from pystruct.models import ChainCRF, BinaryClf
from pystruct.learners import FrankWolfeSSVM, NSlackSSVM

from data_processing import read_file
from feature_extraction import extract_features_scope, extract_features_cue 
from utils import make_splits, get_cue_lexicon, print_cue_lexicons, make_complete_labelarray
from file_writing import *

def cue_detection(C_value, train_file, train_file_parsed, config='training'):
    """ Extract sentence dictionaries, lexicons and features, then train the cue model"""
    sentence_dicts = read_file(train_file, train_file_parsed)
    cue_lexicon, affixal_cue_lexicon = get_cue_lexicon(sentence_dicts)
    sd, instances, labels = extract_features_cue(sentence_dicts, cue_lexicon, affixal_cue_lexicon, config)
    vectorizer = DictVectorizer()
    fvs = vectorizer.fit_transform(instances).toarray()
    model = BinaryClf()
    ssvm = NSlackSSVM(model, C=C_value, batch_size=-1)
    ssvm.fit(fvs, np.asarray(labels))
    return ssvm, vectorizer, cue_lexicon, affixal_cue_lexicon
    
def scope_resolution(C_value, state_value, train_file, train_file_parsed, config='training'):
    """ Extract sentence dictionaries and features, then train the scope model"""
    sentence_dicts = read_file(train_file, train_file_parsed)
    sd, instances, labels, splits = extract_features_scope(sentence_dicts, config)
    vectorizer = DictVectorizer()
    fvs = vectorizer.fit_transform(instances).toarray()
    X_train, y_train = make_splits(fvs, labels, splits)
    model = ChainCRF()
    ssvm = FrankWolfeSSVM(model=model, C=C_value, max_iter=10, random_state=state_value)
    ssvm.fit(X_train, y_train)
    return ssvm, vectorizer

def save_cue_learner(train_file, train_file_parsed):
    """
    Save the cue learner object, the cue vectorizer, the cue lexicon
    and the affixal cue lexicon to files
    """
    cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon = cue_detection(0.20, train_file, train_file_parsed)
    pickle.dump(cue_ssvm, open("cue_model.pkl", "wb"))
    joblib.dump(cue_vectorizer, 'cue_vectorizer.pkl')
    pickle.dump(cue_lexicon, open("cue_lexicon.pkl", "wb"))
    pickle.dump(affixal_cue_lexicon, open("affixal_cue_lexicon.pkl", "wb"))

def save_scope_learner(train_file, train_file_parsed):
    """Save the scope learner object and the scope vectorizer object to files"""
    scope_ssvm, scope_vectorizer = scope_resolution(0.10, 10, train_file, train_file_parsed)
    pickle.dump(scope_ssvm, open("scope_model.pkl", "wb"))
    joblib.dump(scope_vectorizer, 'scope_vectorizer.pkl')

