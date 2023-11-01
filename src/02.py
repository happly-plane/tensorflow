import re

text = open(r"dataset\xiaoshuo.txt","r",encoding="utf-8")
data = text.read() 
data = data.split("\n")
data = [lines for lines in data if len(lines)> 0 ]
data = [lines.strip() for lines in data]

def create_lookup_table(input_data):
        vcoab = set(input_data)
        
        vcoab_to_int = {word: idx for idx, word in enumerate(vcoab)}
        
        int_to_vocab = dict(enumerate(vcoab))
        
        return vcoab_to_int,int_to_vocab
    
x,y = create_lookup_table(data)
print(x)

        
