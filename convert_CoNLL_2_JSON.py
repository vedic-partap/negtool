"""
How to use :
python convert_CoNLL_2_JSON.py <CoNLL output from negation detection> 

"""

import json
import sys

filename = sys.argv[1] 

text = open(filename,'r').read()
d = dict()
d['output']=list(dict())
lines_list =[]
for lines in text.split('\n'):
	if lines!='':
		lines_list.append(lines.split('\t'))
	else:
		sentence =''
		for line in lines_list:
			sentence+=line[1]+' '
		entry = {'sentence':sentence,'negation':[]}
		if len(lines_list)>0:
			count_cues = (len(lines_list[0])-8)/3
			print(count_cues)
			for i in range(int(count_cues)):
				cue =''
				scope=''
				for line in lines_list:
					if(line[7+3*i+0]!='_'):
						cue=line[7+3*i+0]
				for line in lines_list:
					if line[7+3*i+1]!='_':
						scope+=line[7+3*i+1]+' '
				entry['negation'].append({'cue':cue,'scope':scope})
			d['output'].append(entry)
		

		lines_list = []
file = open(filename.split('.')[0]+'.json','w')
json.dump(d,file,indent=4)
		


