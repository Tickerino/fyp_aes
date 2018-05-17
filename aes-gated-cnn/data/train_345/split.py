import random

l = range(1,5043)

random.shuffle(l)

outf = open("train_ids.txt","w")
for x in range(0,2521):
	outf.write(str(l[x]))
	outf.write("\n")
outf.close()

outg = open("dev_ids.txt","w")
for y in range(2521,3782):
	outg.write(str(l[y]))
	outg.write("\n")
outg.close()

outh = open("test_ids.txt","w")
for z in range (3782,5042):
	outh.write(str(l[z]))
	outh.write("\n")
outh.close()
