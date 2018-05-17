import random

l = range(1,6476)

random.shuffle(l)

outf = open("train_ids.txt","w")
for x in range(0,3238):
	outf.write(str(l[x]))
	outf.write("\n")
outf.close()

outg = open("dev_ids.txt","w")
for y in range(3238,4857):
	outg.write(str(l[y]))
	outg.write("\n")
outg.close()

outh = open("test_ids.txt","w")
for z in range (4857,6475):
	outh.write(str(l[z]))
	outh.write("\n")
outh.close()
