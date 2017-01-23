import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from statsmodels.sandbox.stats.runs import mcnemar
from pystruct.models import ChainCRF, BinaryClf
from pystruct.learners import FrankWolfeSSVM, NSlackSSVM

from data_processing import read_file
from feature_extraction import extract_features_scope, extract_features_cue 
from utils import make_splits, get_cue_lexicon, print_cue_lexicons, make_complete_labelarray
from evaluation import *

def cue_detection(C_value, train_file, train_file_parsed, dev_file, dev_file_parsed, config='training'):
    sentence_dicts = read_file(train_file, train_file_parsed)
    cue_lexicon, affixal_cue_lexicon = get_cue_lexicon(sentence_dicts)
    sd, instances, labels = extract_features_cue(sentence_dicts, cue_lexicon, affixal_cue_lexicon, config)
    vectorizer = DictVectorizer()
    fvs = vectorizer.fit_transform(instances).toarray()
    model = BinaryClf()
    ssvm = NSlackSSVM(model, C=C_value, batch_size=-1)
    ssvm.fit(fvs, np.asarray(labels))

    dev_sentence_dicts = read_file(dev_file, dev_file_parsed)
    dev_sd, dev_instances, dev_labels = extract_features_cue(dev_sentence_dicts, cue_lexicon, affixal_cue_lexicon, config)
    dev_fvs = vectorizer.transform(dev_instances).toarray()
    y_pred = ssvm.predict(dev_fvs)
    dev_y_pred = make_complete_labelarray(dev_sentence_dicts, y_pred)
    dev_y_labels = make_complete_labelarray(dev_sentence_dicts, dev_labels)
    convert_cues_to_fileformat(dev_sentence_dicts, dev_y_pred, affixal_cue_lexicon, "../data/gold/cde.txt")
    
def scope_resolution(C_value, train_file, train_file_parsed, dev_file, dev_file_parsed, state_value, config='training'):
    sentence_dicts = read_file(train_file, train_file_parsed)
    sd, instances, labels, splits = extract_features_scope(sentence_dicts, "../data/gold/cdtd.txt", "../data/cdtd_parsed.txt", config)
    vectorizer = DictVectorizer()
    fvs = vectorizer.fit_transform(instances).toarray()
    X_train, y_train = make_splits(fvs, labels, splits)
    
    start = time.clock()  
    model = ChainCRF()
    ssvm = FrankWolfeSSVM(model=model, C=C_value, max_iter=10, random_state=state_value)
    ssvm.fit(X_train, y_train)

    dev_sentence_dicts = read_file("system_cdd_cues.txt", "../data/cde_parsed2.txt")
    sentences, dev_instances, dev_labels, dev_splits = extract_features_scope(dev_sentence_dicts, "system_cdd_cues.txt", "../data/cde_parsed2.txt", config)
    dev_fvs = vectorizer.transform(dev_instances).toarray()
    X_dev, y_dev = make_splits(dev_fvs, dev_labels, dev_splits)
    y_pred = ssvm.predict(X_dev)
    end = time.clock()
    convert_list_to_fileformat(sentences, y_pred)
    print "Time elapsed:", end-start
    #sanity_check("../data/gold/cdd.txt", "system_cdd.txt")

    #return y_pred, y_dev

def compare_statistics():
    """
    y1, y_gold = scope_resolution(0.35, 10, 'A')
    #print y_gold.shape
    y2 = scope_resolution(0.30, 10, 'B')[0]
    significance_test_scope(y1,y2, y_gold)
    """

    y1, y_gold1 = cue_detection(0.40, 'A')
    y2, y_gold2 = cue_detection(0.23, 'B')
    if y_gold1 != y_gold2:
        print "Noe er veldig galt"
        return
    significance_test_cue(y1, y2, y_gold1)
    
def significance_test_scope(y1, y2, y_gold):
    s1 = []
    s2 = []
    for i in range(len(y1)):
        y1[i] = convert_to_IO(y1[i])
        y2[i] = convert_to_IO(y2[i])
        y_gold[i] = convert_to_IO(y_gold[i])
        for j in range(len(y1[i])):
            if y1[i][j] == y_gold[i][j]:
                s1.append(1)
            else:
                s1.append(0)

            if y2[i][j] == y_gold[i][j]:
                s2.append(1)
            else:
                s2.append(0)
        
    print "Number of samples:", len(s1)
    test = mcnemar(s1, s2)
    print "P-value from McNemar-test:", test[1]

def significance_test_cue(y1, y2, y_gold):
    s1 = []
    s2 = []
    for i in range(len(y1)):
        if y1[i] == y_gold[i]:
            s1.append(1)
        else:
            s1.append(0)
        if y2[i] == y_gold[i]:
            s2.append(1)
        else:
            s2.append(0)

    print "Number of samples:", len(s1)
    test = mcnemar(s1, s2)
    print "P-value from McNemar test:", test[1]

