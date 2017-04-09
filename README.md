# negtool
A tool for detecting negation cues and scopes in natural language text, as described in the paper *An open-source tool for negation detection: a maximum-margin approach* ([pdf](http://www.aclweb.org/anthology/W/W17/W17-1810.pdf)) by M. Enger, E. Velldal \& L. Øvrelid, presented at the 2017 [SemBEaR](http://www.cse.unt.edu/sembear2017/) workshop ([slides](http://www.velldal.net/erik/talks/sembear2017.pdf)). 

## Dependencies
In order to use negtool, the following libraries need to be installed:
   * numpy
   * scikit-learn
   * pystruct
   * networkx

## Running the tool with raw text
To run negtool with raw text, you need to have CoreNLP installed. Run the tool with the command

`python negtool.py -m raw -f <inputfile> -d <absolute path to corenlp>`

## Running the tool with parsed text
To run the tool with a parsed input file, the format of the file needs to be CoNLL_X, with the following information encoded: 

   * Column 1: token index
   * Column 2: token
   * Column 3: lemma
   * Column 5: PoS-tag
   * Column 7: head index
   * Column 8: dependency relation

Run the tool with the command

`python negtool.py -m parsed -f <inputfile>`

## Output

The output is a file where the first 8 columns are identical to the inputfile, and the following columns include cues and scopes encoded in the CD format. The column for events is included with the symbol "_". The name of the file is the same as the name of the input file, but with the extension `.neg` instead of the original file extension. 

## Training your own models
You can train your own cue and/or scope model with a new dataset. The dataset needs to be on CoNLL_X format with cues and scopes encoded on the CD format starting at column 9. Training is done by running

`python train.py -m <model to train> -tf <training file> -cp <cue regularisation> -sp <scope regularisation>`

For the -m option, the program accepts either cue, scope or all. The default value for both the cue regularisation parameter and the scope regularisation parameter is 0.20. 
## Evaluating the models
Our provided models and your own trained models can be evaluated with the 2012 *SEM evaluation script which can be found [here](http://www.clips.ua.ac.be/sem2012-st-neg/data.html). In order to get a file with predictions that can be evaluated with that script, you can run the program `evaluation.py`:

`python evaluation.py -cm <cue model> -sm <scope model> -cl <cue lexicon> -acl <affixal cue lexicon> -cv <cue vectorizer> -sv <scope vectorizer> -tf <testfile> -cdf <parsed CD testfile> -ff <file format> -e2e <end to end/gold cues>`

If you want to evaluate Conan Doyle (CD) files, you need to provide a parsed version of the file with dependency relations. If you evaluate Conll-X files, this is not necessary. 

The file format argument accepts either "CD" for Conan Doyle files or "parsed" for Conll-X files. 

If you want to evaluate end-to-end predictions, set the -e2e argument to true. If you want to evaluate scopes with gold cues, set the -e2e argument to false. 

The output will be a file on the CD format (for every column, not just the cue columns) named "<input filename>_evaluation_scopes.txt". 

## Citing
Please cite the following paper if you use the tool:
```
@InProceedings{Enger17,
  author    = {Enger, Martine  and  Velldal, Erik  and  {\O}vrelid, Lilja},
  title     = {An open-source tool for negation detection: a maximum-margin approach},
  booktitle = {Proceedings of the Workshop Computational Semantics Beyond Events and Roles},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {64--69}
}
```
