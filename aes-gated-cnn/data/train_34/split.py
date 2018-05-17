import random

l = range(1,3496)

random.shuffle(l)

outf = open("train_ids.txt","w")
for x in range(0,1748):
	outf.write(str(l[x]))
	outf.write("\n")
outf.close()

outg = open("dev_ids.txt","w")
for y in range(1748,2622):
	outg.write(str(l[y]))
	outg.write("\n")
outg.close()

outh = open("test_ids.txt","w")
for z in range (2622,3495):
	outh.write(str(l[z]))
	outh.write("\n")
outh.close()
