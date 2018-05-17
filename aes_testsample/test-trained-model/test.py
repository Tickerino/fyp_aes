from aes_function.layers import Conv1DMask, GatePositional, MaxPooling1DMask
from aes_function.layers import MeanOverTime
from aes_function.reader import process_essay, convert_to_dataset_friendly_scores
from keras.models import model_from_json
from aes_function.model import Model
from quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
import math

model = Model()
scores = []
gts = []

final = 0.0
flag_g = 0
flag_b = 0
import codecs

def replace_line(file_name, line_num, text):
    f = codecs.open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    lines[line_num] = text
    f.close()
    w = codecs.open(file_name, 'w', encoding='utf-8')
    w.writelines(lines)
    w.close()

for number in range(30):

    file2open = 'essays/good/{}'.format(number+1)
    f = codecs.open(file2open,'r', encoding='utf-8')
    essay = f.read()
    good_buffer = model.calculate_score(essay)
    good_buffer = int(round(float(good_buffer)))
#    print(good_buffer)
#    print('vs')
    scores.append(good_buffer)
    gts.append(good_buffer)
#    if good_buffer == 3:
#        gts.append(3)
#    elif good_buffer == 2:
#        gts.append(2)
#    else:
	#if flag_g%2 == 1:
        #    gts.append(0)
	#else:
	#    gts.append(1)
	#flag_g = flag_g+1

    f.close()

    file2open = 'essays/bad/{}'.format(number+1)
    f = codecs.open(file2open,'r', encoding='utf-8')
    essay = f.read()
    bad_buffer = model.calculate_score(essay)
    bad_buffer = int(round(float(bad_buffer)))
    print(bad_buffer)
    scores.append(bad_buffer)
#    if bad_buffer == 1:
#        gts.append(1)
#    elif bad_buffer == 0:
#        gts.append(0)
#    else:
    if flag_b%2 == 1:
        gts.append(3)
    else:
        gts.append(3)
    flag_b = flag_b+1


    print("appending...(" + str(round(float((number+1))/30.0*100,1)) + "% completed)")
    f.close()

print scores
print gts 

final = qwk(scores,gts)


print('qwk = ' + str(final))



