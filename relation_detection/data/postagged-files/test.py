import re

temp = "Syrian_JJ President_NNP Travels_NNP To_TO Egypt_VB"
temp = ' '+temp
print(re.sub(r' [^_]+_', ' ', temp))