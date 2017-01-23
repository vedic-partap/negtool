import itertools
from utils import get_affix_cue, count_multiword_cues, check_mw_start, convert_to_IO
import numpy as np

def convert_cues_to_fileformat(sentences, labels, affix_cue_lexicon, filename, mode):
    """
    Write the predicted cues to file, using the CD format. 
    """
    infile = open(filename, "r")
    output_filename = filename.split(".")[0] + "_cues.neg"
    outfile = open(output_filename, "w")
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
            for i in range(upper_limit): #NB change to 7 for CD-dataset
                outfile.write("%s\t" %tokens[i])
            if n_cues == 0:
                outfile.write("***\n")
            else:
                for cue_i in range(n_cues):
                    if labels[sent_counter][line_counter] < 0:
                        outfile.write("_\t_\t_\t")
                    else:
                        #cue-line
                        if written_cues[cue_i] or written_cue_on_line:
                            outfile.write("_\t_\t_\t")
                        else:
                            #NB change to tokens[3] for CD-dataset
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

def convert_scopes_to_fileformat(sentences, labels, filename, mode):
    """
    Write predicted scopes to file, using the CD format
    """
    filename_base = filename.split("_cues.neg")[0]
    output_filename = filename_base + ".neg"
    infile = open(filename, "r")
    outfile = open(output_filename, "w")
    sent_counter = 0
    line_counter = 0
    scope_counter = 0
    upper_limit = 7 if mode == "raw" else 8    
    n_cues = 0
    for line in infile:
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
            for i in range(upper_limit):
                outfile.write("%s\t" %tokens[i])
            for cue_i in range(n_cues):
                outfile.write("%s\t" %tokens[upper_limit + 3*cue_i]) #write gold cue
                #write scope
                if labels[scope_counter][line_counter] == 0 or labels[scope_counter][line_counter] == 2 or (labels[scope_counter][line_counter] == 3 and cues[cue_i][2] == 'a'):
                    if cues[cue_i][2] == 'a' and sent[int(cues[cue_i][1])][3] == tokens[1]:
                        outfile.write("%s\t" %(tokens[1].replace(cues[cue_i][0], "")))
                    elif tokens[upper_limit + 3*cue_i] != "_":
                        outfile.write("_\t")
                    else:
                        outfile.write("%s\t" %tokens[1])
                else:
                    outfile.write("_\t")

                outfile.write("%s\t" %tokens[upper_limit + 2 + 3*cue_i]) #write gold event
                scope_counter += 1
            
            scope_counter -= n_cues
            line_counter += 1
            outfile.write("\n")

    infile.close()
    outfile.close()
            
