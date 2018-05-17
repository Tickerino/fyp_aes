import csv

main = []
with open('training_set_rel3_150v.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for r in reader:
        main.append(r)
header = main[0]
print(header)
essays = main[1:]
for set_type in ['train', 'test', 'dev']:
    file2open = 'train_150/{}_ids.txt'.format(set_type)
    selected = [header]
    with open(file2open, 'r') as f:
        content = f.readlines()
        content = [x.rstrip('\n') for x in content]
        for essay in essays:
            essay_id = essay[0]
            if (essay_id in content):
                selected.append(essay)
    with open('train_150/{}.tsv'.format(set_type), 'w+') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(selected)
    print('Written data_150/{}.tsv'.format(set_type))
