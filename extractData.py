import pandas as pd

data = pd.read_csv("linked-extractions.tsv", sep='\t', header=None)

freebase_id = data[3]
file = open("Base.sql", 'w')

i = 0
while i < len(freebase_id):
	temp = 'INSERT INTO Base (entity, relation, value, freebase_id, freebase_entity, link_scroe, link_am_score) VALUES ("'+str(data[0][i])+'", "'+str(data[1][i])+'", "'+str(data[2][i])+'", "'+str(data[3][i])+'", "'+str(data[4][i])+'", "'+str(data[5][i])+'", "'+str(data[6][i])+'");'
	file.write(temp + '\n')
	i = i + 1



