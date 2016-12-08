from pystruct.utils import SaveLogger
import pickle
import sys
import argparse
import os.path
from sklearn.externals import joblib

from data_processing import read_file
from input_output import read_parsed_data, read_cuepredicted_data
from feature_extraction import extract_features_scope, extract_features_cue 
from parse_raw_files import run_corenlp
from utils import *
from evaluation import *

def load_cue_learner():
    """
    Loads the object containing the cue learner from file
    """

    cue_ssvm = pickle.load(open("cue_model.pkl", "rb"))
    cue_vectorizer = joblib.load('cue_vectorizer.pkl')
    cue_lexicon = pickle.load(open("cue_lexicon.pkl", "rb"))
    affixal_cue_lexicon = pickle.load(open("affixal_cue_lexicon.pkl", "rb"))
    return cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon

def load_scope_learner():
    """
    Loads the object containing the scope learner from file
    """

    scope_ssvm = pickle.load(open("scope_model.pkl", "rb"))
    scope_vectorizer = joblib.load("scope_vectorizer.pkl")
    return scope_ssvm, scope_vectorizer

def run_cue_learner(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, filename, mode):
    """
    Reads the file with the input data, extracts features for cue detection, does cue prediction and converts the predicted cues to the CD file format
    """

    #dev_sentence_dicts = read_file("../data/gold/cdd.txt", "../data/cdd_parsed.txt")
    dev_sentence_dicts = read_parsed_data(filename, mode)
    dev_sd, dev_instances = extract_features_cue(dev_sentence_dicts, cue_lexicon, affixal_cue_lexicon, 'prediction')
    dev_fvs = cue_vectorizer.transform(dev_instances).toarray()
    y_pred = cue_ssvm.predict(dev_fvs)
    dev_y_pred = make_complete_labelarray(dev_sentence_dicts, y_pred)
    convert_cues_to_fileformat(dev_sentence_dicts, dev_y_pred, affixal_cue_lexicon, filename, mode)

def run_scope_learner(scope_ssvm, scope_vectorizer, mode):
    """
    Reads the file with the predicted cues, extracts features for scope resolution, does scope prediction and converts the predicted scopes for the predicted cues to the CD file format
    """

    sentence_dicts = read_cuepredicted_data("system_cdd_cues.txt", mode)
    sentences, dev_instances, dev_splits = extract_features_scope(sentence_dicts, "system_cdd_cues.txt", "../data/cdd_parsed.txt", 'prediction')
    dev_fvs = scope_vectorizer.transform(dev_instances).toarray()
    print "Number of sentences:", len(sentences)
    X_dev, y_dev = make_splits(dev_fvs, [], dev_splits)
    y_pred = scope_ssvm.predict(X_dev)
    convert_scopes_to_fileformat(sentences, y_pred, mode)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--mode', help="program mode. either raw or parsed", type=str, choices=['raw','parsed'])
    argparser.add_argument('-f', '--filename', help="input file", type=str)
    argparser.add_argument('-d', '--directory', help="absolute path to corenlp directory. needs to be provided in raw mode", type=str, nargs='?')
    args = argparser.parse_args()
    print "Fil:", args.filename
    print "Mode:", args.mode

    #save_cue_learner()
    filename = args.filename
    if not os.path.isfile(filename):
        print "ERROR: File does not exist. Program will exit"
        sys.exit(1)
    if args.mode == 'raw':
        print "input dir:", args.directory
        path_to_corenlp = args.directory
        if args.directory == None:
            path_to_corenlp = raw_input("Absolute path to CoreNLP directory:")
        elif not os.path.exists(args.directory):
            path_to_corenlp = raw_input("ERROR: You specified the wrong path. Please specify the right path: ")
        run_corenlp(path_to_corenlp, args.filename)
        filename = args.filename + ".conll"
    cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon = load_cue_learner()
    run_cue_learner(cue_ssvm, cue_vectorizer, cue_lexicon, affixal_cue_lexicon, filename, args.mode)

    #save_scope_learner()
    scope_ssvm, scope_vectorizer = load_scope_learner()
    run_scope_learner(scope_ssvm, scope_vectorizer, args.mode)
