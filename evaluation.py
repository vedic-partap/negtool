import itertools
from statsmodels.sandbox.stats.runs import mcnemar
from statsmodels.stats.descriptivestats import sign_test
from utils import get_affix_cue, count_multiword_cues, check_mw_start, convert_to_IO
import numpy as np

def convert_cues_to_fileformat(sentences, labels, affix_cue_lexicon, filename, mode):
    infile = open(filename, "r")
    outfile = open("system_cdd_cues.txt", "w")
    sent_counter = 0
    line_counter = 0
    upper_limit = 7 if mode == "raw" else 8
    n_cues = sum(i > 0 for i in labels[sent_counter])
    n_mwc, has_mwc = count_multiword_cues(sentences[sent_counter], labels[sent_counter])
    if has_mwc:
        n_cues += n_mwc - 1
    written_cues = n_cues*[False]
    for line in infile:
        tokens = line.split()
        if len(tokens) == 0:
            sent_counter += 1
            line_counter = 0
            if sent_counter < len(labels):
                n_cues = sum(i > 0 for i in labels[sent_counter])
                n_mwc, has_mwc = count_multiword_cues(sentences[sent_counter], labels[sent_counter])
                if has_mwc:
                    n_cues += n_mwc - 1
                written_cues = n_cues*[False]
            outfile.write("\n")
        else:
            written_cue_on_line = False
            for i in range(upper_limit): #NB ma byttes til 7 for cdd
                outfile.write("%s\t" %tokens[i])
            if n_cues == 0:
                outfile.write("***\n")
            else:
                for cue_i in range(n_cues):
                    if labels[sent_counter][line_counter] < 0:
                        outfile.write("_\t_\t_\t")
                    else:
                        #cue-linje.
                        if written_cues[cue_i] or written_cue_on_line:
                            outfile.write("_\t_\t_\t")
                        else:
                            #NB endre tilbake til tokens[3] for cdd
                            affix = get_affix_cue(tokens[1].lower(), affix_cue_lexicon)
                            if affix != None:
                                outfile.write("%s\t_\t_\t" %affix)
                                written_cues[cue_i] = True
                            else:
                                outfile.write("%s\t_\t_\t" %tokens[1])
                                prev_token = sentences[sent_counter][line_counter-1][3].lower() if line_counter > 0 else 'null'
                                if not check_mw_start(tokens[1].lower(), prev_token):
                                    written_cues[cue_i] = True
                            written_cue_on_line = True
                line_counter += 1
                outfile.write("\n")
    infile.close()
    outfile.close()

def convert_scopes_to_fileformat(sentences, labels, mode):
    #infile = open("../data/gold/cde.txt", "r")
    infile2 = open("system_cdd_cues.txt", "r")
    outfile = open("system_cdd.txt", "w")
    sent_counter = 0
    line_counter = 0
    scope_counter = 0
    upper_limit = 7 if mode == "raw" else 8    
    n_cues = 0
    for line in infile2:
        tokens = line.split()
        if len(tokens) == 0:
            sent_counter += 1
            scope_counter += n_cues
            line_counter = 0
            n_cues = 0
            outfile.write("\n")
        elif tokens[-1] == "***":
            outfile.write(line)
        else:
            sent = sentences[sent_counter]
            cues = sent['cues']
            n_cues = len(cues)
            for i in range(upper_limit): #NB endre til 7 for cdd
                outfile.write("%s\t" %tokens[i])
            for cue_i in range(n_cues):
                outfile.write("%s\t" %tokens[upper_limit + 3*cue_i]) #skriver gull-cue. 7 for cdd
                #skriver scope. nb tokens[3] for cdd
                if labels[scope_counter][line_counter] == 0 or labels[scope_counter][line_counter] == 2 or (labels[scope_counter][line_counter] == 3 and cues[cue_i][2] == 'a'):
                    if cues[cue_i][2] == 'a' and sent[int(cues[cue_i][1])][3] == tokens[1]:
                        outfile.write("%s\t" %(tokens[1].replace(cues[cue_i][0], ""))) #nb tokens[3] for cdd
                    elif tokens[upper_limit + 3*cue_i] != "_": #nb endre til 7 for cdd
                        outfile.write("_\t")
                    else:
                        outfile.write("%s\t" %tokens[1])
                else:
                    outfile.write("_\t")

                outfile.write("%s\t" %tokens[upper_limit + 2 + 3*cue_i]) #skrive event. 9 for cdd
                scope_counter += 1
            
            scope_counter -= n_cues
            line_counter += 1
            outfile.write("\n")

    infile2.close()
    outfile.close()

def sanity_check(gold, system):
    with open(gold, 'r') as gold_file, open(system,'r') as sys_file:
        line_counter = 0
        i = 0
        for line in gold_file:
            sys_line = sys_file.readline().split()
            gold_line = line.split()
            if sys_line != gold_line:
                if i < 30:
                    print "Line number:", line_counter
                i += 1
            if len(gold_line) == 0 and i > 20 and i < 30:
                print "\n"
            line_counter += 1

def compute_baseline():
    infile = open("../data/gold/cde.txt", "r")
    outfile = open("baseline_cdd.txt", "w")
    n_cues = -1
    for line in infile:
        tokens = line.split()
        if len(tokens) == 0:
            outfile.write("\n")
        elif tokens[-1] == "***":
            outfile.write(line)
        else:
            n_cues = (len(tokens) - 9)/3 + 1
            for i in range(7):
                outfile.write("%s\t" %tokens[i])

            for cue_i in range(n_cues):
                outfile.write("%s\t" %tokens[7 + 3*cue_i]) #skriver gull-cue
                if tokens[7+ 3*cue_i] == "_":
                    outfile.write("%s\t" %tokens[3]) #alle tokens i setningen er in-scope unntatt cue
                else:
                    outfile.write("_\t")
                outfile.write("%s\t" %tokens[9 + 3*cue_i]) #skriver event
            outfile.write("\n")

    infile.close()
    outfile.close()

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
    #test = sign_test(np.array(s1)-np.array(s2))
    print "P-value from McNemar test:", test[1]

if __name__ == "__main__":
    compute_baseline()
            
