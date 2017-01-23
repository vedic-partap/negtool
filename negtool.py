from pystruct.utils import SaveLogger
import pickle
import sys
import argparse
import os.path
from sklearn.externals import joblib

from input_output import *
from feature_extraction import extract_features_scope, extract_features_cue 
from utils import *
from evaluation import *

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

def save_cue_learner():
    """
    Saves the cue learner object, the cue vectorizer, the cue lexicon
    and the affixal cue lexicon to files
    """
    cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon = cue_detection(0.20)
    pickle.dump(cue_ssvm, open("cue_model.pkl", "wb"))
    joblib.dump(cue_vectorizer, 'cue_vectorizer.pkl')
    pickle.dump(cue_lexicon, open("cue_lexicon.pkl", "wb"))
    pickle.dump(affixal_cue_lexicon, open("affixal_cue_lexicon.pkl", "wb"))

def save_scope_learner():
    """Saves the scope learner object and the scope vectorizer object to files"""
    scope_ssvm, scope_vectorizer = scope_resolution(0.10,10)
    pickle.dump(scope_ssvm, open("scope_model.pkl", "wb"))
    joblib.dump(scope_vectorizer, 'scope_vectorizer.pkl')

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
    print "Number of sentences:", len(sentences)
    X_dev, y_dev = make_splits(dev_fvs, [], dev_splits)
    y_pred = scope_ssvm.predict(X_dev)
    convert_scopes_to_fileformat(sentences, y_pred, filename, mode)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--mode', help="program mode. either raw or parsed", type=str, choices=['raw','parsed'])
    argparser.add_argument('-f', '--filename', help="input file", type=str)
    argparser.add_argument('-d', '--directory', help="absolute path to corenlp directory. needs to be provided in raw mode", type=str, nargs='?')
    args = argparser.parse_args()

    filename = args.filename
    if not os.path.isfile(filename):
        print "ERROR: File does not exist. Program will exit"
        sys.exit(1)
    if args.mode == 'raw':
        path_to_corenlp = args.directory
        if args.directory == None:
            path_to_corenlp = raw_input("Absolute path to CoreNLP directory:")
        elif not os.path.exists(args.directory):
            path_to_corenlp = raw_input("ERROR: You specified the wrong path. Please specify the right path: ")
        run_corenlp(path_to_corenlp, args.filename)
        filename = args.filename + ".conll"
    cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon = load_cue_learner()
    run_cue_learner(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, filename, args.mode)
    cue_file = filename.split(".")[0] + "_cues.neg"
    scope_ssvm, scope_vectorizer = load_scope_learner()
    run_scope_learner(scope_ssvm, scope_vectorizer, cue_file, args.mode)
