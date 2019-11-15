dataset = open()
out = open()

for line in dataset:
	if line.strip():
		token,entity,vector = line.split("\t")
			##print(line.split('\t'))
		featSet = vector.rstrip().split(',')
		if len(featSet) < 46:
				#print(" se tiene que cambiar")
				fix_vector = line.replace(',,',',___,___,')
				out.write(line)
			else:
				fix_vector = vector.replace(',,',',___,')
				out.write(line)
