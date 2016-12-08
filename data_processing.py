import numpy as np

def read_file(filename, conll_filename):
    """
    Read input file and make dictionaries for each sentence. 
    Used for training with the CD dataset.
    """
    with open(filename, 'r') as infile1, open(conll_filename) as infile2:
        sentence = {}
        cues = []
        mw_cues = []
        scopes = {}
        events = {}
        line_counter = 0
        counter = 0
        cue_counter = 0
        prev_cue_column = -1
        instances = []

        for line in infile1:
            conll_line = infile2.readline()
            token_dict = {}
            tokens = line.split()
            conll_tokens = conll_line.split()
            #check for sentence end
            if len(tokens) == 0:
                for key in sentence:
                    head_index = int(sentence[key]['head']) - 1
                    if head_index > -1:
                        sentence[key]['head-pos'] = sentence[head_index][5]
                    else:
                        sentence[key]['head-pos'] = sentence[key][5]

                if(len(scopes) != len(cues)):
                    for i in range(len(cues)):
                        if not i in scopes:
                            scopes[i] = []

                sentence['cues'] = cues
                sentence['mw_cues'] = mw_cues
                sentence['scopes'] = scopes
                sentence['events'] = events
                    
                if len(cues) > 0:
                    sentence['neg'] = True
                else:
                    sentence['neg'] = False

                #yield sentence
                instances.append(sentence)
                sentence = {}
                counter = 0
                prev_cue_column = -1
                cues = []
                mw_cues = []
                scopes = {}
                events = {}
                line_counter += 1
                continue

            for i in range(len(tokens)):            
                if tokens[i] != "_" and  i < 6:
                    token_dict[i] = tokens[i]
                elif tokens[i] != "***" and tokens[i] != "_" and i > 6 and (i-1) % 3 == 0:
                    if i == prev_cue_column:
                        cues[-1][2] = 'm'
                        prev_cue_column = i
                        if cues[-1][2] == 'm':
                            mw_cues.append([cues[-1][0],cues[-1][1]])
                        mw_cues.append([tokens[i], counter])
                    elif tokens[i] != tokens[3]:
                        cues.append([tokens[i], counter, 'a'])
                        prev_cue_column = i
                    else:
                        cues.append([tokens[i], counter, 's'])
                        prev_cue_column = i
                elif tokens[i] != "***" and tokens[i] != "_" and i > 6 and (i-2) % 3 == 0:
                    cue_counter = (i-8)/3
                    if cue_counter in scopes:
                        scopes[cue_counter].append([tokens[i], counter])
                    else:
                        scopes[cue_counter] = [[tokens[i], counter]]
                elif tokens[i] != "***" and tokens[i] != "_" and i > 6 and (i-3) % 3 == 0:
                    cue_counter = (i-9)/3
                    events[cue_counter] = tokens[i]
            token_dict['head'] = conll_tokens[6]
            token_dict['deprel'] = conll_tokens[7]
            sentence[counter] = token_dict
            counter += 1
            line_counter += 1
        return instances

if __name__ == '__main__':
    cue_counter = 0
    scope_counter = 0
    event_counter = 0
    sentence_counter = 0
    negsent_counter = 0
    ex_sent = None
    for sentence in read_file("../data/gold/cdd.txt", "../data/cdd_parsed.txt"):
        cue_counter += len(sentence['cues'])
        scope_counter += len(sentence['scopes'])
        event_counter += len(sentence['events'])
        sentence_counter += 1
        if sentence['neg']:
           negsent_counter += 1

    print "Antall setninger:", sentence_counter
    print "Antall negerte setninger:", negsent_counter
    print "Antall cues:", cue_counter
    print "Antall scopes:", scope_counter
    print "Antall events:", event_counter

