# negtool
A tool for detecting negation cues and scopes in natural language text.

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

The output is a file with cues and scopes encoded in the CD format. 
