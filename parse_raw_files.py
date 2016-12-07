import subprocess
import sys

def run_corenlp(corenlp_path, filename):
    with open(filename, 'r') as infile:
        line1 = infile.readline()
        line2 = infile.readline()
        if line1.split()[0] == '1' and line2.split()[0] == '2':
            print "ERROR: File seems to be parsed already"
            sys.exit(1)
            
    absolute_path = corenlp_path + "/*"
    args = ['java', '-cp', absolute_path, '-Xmx1800m', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',  '-annotators', 'tokenize,ssplit,pos,lemma,depparse', '-file', filename, '-outputFormat', 'conll']
    pipe = subprocess.call(args)
    
    
