import pickle
import sys
import argparse
from os import remove, path
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from pystruct.models import ChainCRF, BinaryClf
from pystruct.learners import FrankWolfeSSVM, NSlackSSVM
from pystruct.utils import SaveLogger

from file_reading import *
from feature_extraction import extract_features_scope, extract_features_cue 
from utils import *
from file_writing import *
from read_labelled_data import read_file

def load_cue_learner():
    """
    Loads the object containing the cue learner from file
    Returns the cue learner, the cue vectorizer, the cue lexicon and the
    affixal cue lexicon
    """
    cue_ssvm = pickle.load(open("objectfiles/cue_model.pkl", "rb"))
    cue_vectorizer = joblib.load("objectfiles/cue_vectorizer.pkl")
    cue_lexicon = pickle.load(open("objectfiles/cue_lexicon.pkl", "rb"))
    affixal_cue_lexicon = pickle.load(open("objectfiles/affixal_cue_lexicon.pkl", "rb"))
    return cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon

def load_scope_learner():
    """
    Loads the object containing the scope learner from file
    Returns the scope learner and the scope vectorizer
    """
    scope_ssvm = pickle.load(open("objectfiles/scope_model.pkl", "rb"))
    scope_vectorizer = joblib.load("objectfiles/scope_vectorizer.pkl")
    return scope_ssvm, scope_vectorizer

def train_cue_learner(sentence_dicts, C_value):
    cue_lexicon, affixal_cue_lexicon = get_cue_lexicon(sentence_dicts)
    cue_sentence_dicts, cue_instances, cue_labels = extract_features_cue(sentence_dicts, cue_lexicon, affixal_cue_lexicon, 'training')
    vectorizer = DictVectorizer()
    fvs = vectorizer.fit_transform(cue_instances).toarray()
    model = BinaryClf()
    cue_ssvm = NSlackSSVM(model, C=C_value, batch_size=-1)
    cue_ssvm.fit(fvs, np.asarray(cue_labels))
    return cue_ssvm, vectorizer, cue_lexicon, affixal_cue_lexicon

def train_scope_learner(sentence_dicts, C_value):
    scope_sentence_dicts, scope_instances, scope_labels, sentence_splits = extract_features_scope(sentence_dicts, 'training')
    vectorizer = DictVectorizer()
    fvs = vectorizer.fit_transform(scope_instances).toarray()
    X_train, y_train = make_splits(fvs, scope_labels, sentence_splits)
    model = ChainCRF()
    scope_ssvm = FrankWolfeSSVM(model=model, C=C_value, max_iter=10)
    scope_ssvm.fit(X_train, y_train)
    return scope_ssvm, vectorizer

def run_cue_learner(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, filename, mode):
    """
    Reads the file with the input data, extracts features for cue detection,
    does cue prediction and converts the predicted cues to the CD file format
    """
    dev_sentence_dicts = read_parsed_data(filename, mode)
    dev_sd, dev_instances = extract_features_cue(dev_sentence_dicts, cue_lexicon, affixal_cue_lexicon, 'prediction')
    dev_fvs = cue_vectorizer.transform(dev_instances).toarray()
    y_pred = cue_ssvm.predict(dev_fvs)
    dev_y_pred = make_complete_labelarray(dev_sentence_dicts, y_pred)
    convert_cues_to_fileformat(dev_sentence_dicts, dev_y_pred, affixal_cue_lexicon, filename, mode)

def run_scope_learner(scope_ssvm, scope_vectorizer, filename, mode):
    """
    Reads the file with the predicted cues, extracts features for scope resolution,
    does scope prediction and converts the predicted scopes for the predicted cues
    to the CD file format
    """
    sentence_dicts = read_cuepredicted_data(filename, mode)
    sentences, dev_instances, dev_splits = extract_features_scope(sentence_dicts, 'prediction')
    dev_fvs = scope_vectorizer.transform(dev_instances).toarray()
    X_dev, y_dev = make_splits(dev_fvs, [], dev_splits)
    y_pred = scope_ssvm.predict(X_dev)
    convert_scopes_to_fileformat(sentences, y_pred, filename, mode)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--mode', help="program mode. either raw or parsed or retraining", type=str, choices=['raw','parsed', 'retraining'])
    argparser.add_argument('-f', '--filename', help="input file", type=str, nargs='?')
    argparser.add_argument('-d', '--directory', help="absolute path to corenlp directory. needs to be provided in raw mode", type=str, nargs='?')
    args = argparser.parse_args()

    if args.mode == 'retraining':
        training_file = raw_input("Enter file name of training file: ")
        test_file = raw_input("Enter file name of test file: ")
        sentence_dicts = read_file(training_file)
        print "Setning 0:"
        print sentence_dicts[0]['cues']
        print sentence_dicts[0]['scopes']
        print ""
        print "Setning 1:"
        print sentence_dicts[1]['cues']
        print sentence_dicts[1]['scopes']
        print ""
        print "Setning 2:"
        print sentence_dicts[2]['cues']
        print sentence_dicts[2]['scopes']
        cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon = train_cue_learner(sentence_dicts, 0.20)
        print affixal_cue_lexicon
        run_cue_learner(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, test_file, 'parsed')

        scope_ssvm, scope_vectorizer = train_scope_learner(sentence_dicts, 0.20)
        cue_file = test_file.split(".")[0] + "_cues.neg"
        run_scope_learner(scope_ssvm, scope_vectorizer, cue_file, 'parsed')
        remove(cue_file)
    else:
        filename = args.filename
        if not path.isfile(filename):
            print "ERROR: File does not exist. Program will exit"
            sys.exit(1)
        if args.mode == 'raw':
            path_to_corenlp = args.directory
            if args.directory == None:
                path_to_corenlp = raw_input("Absolute path to CoreNLP directory:")
            elif not path.exists(args.directory):
                path_to_corenlp = raw_input("ERROR: You specified the wrong path. Please specify the right path:")
                run_corenlp(path_to_corenlp, args.filename)
            filename = args.filename + ".conll"

        cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon = load_cue_learner()
        run_cue_learner(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, filename, args.mode)
        cue_file = filename.split(".")[0] + "_cues.neg"
        scope_ssvm, scope_vectorizer = load_scope_learner()
        run_scope_learner(scope_ssvm, scope_vectorizer, cue_file, args.mode)
        remove(cue_file)
