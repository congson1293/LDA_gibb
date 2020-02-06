import os
import utils
import config
import unicodedata
from io import open
import time
import sys
import preprocessing



def load_data(dataset):
    documents = []
    total_files = 0
    total_time = 0
    stack = os.listdir(dataset)
    print 'loading data in ' + dataset
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = dataset + '/' + file_name
        if (os.path.isdir(file_path)):  # neu la thu muc thi day vao strong stack
            utils.push_data_to_stack(stack, file_path, file_name)
        else:
            begin = time.time()
            try:
                with open(file_path, 'r', encoding='utf-8') as fp:
                    content = preprocessing.preprocessing(fp.read())
                    documents.append(content)
            except:
                with open(file_path, 'r', encoding='utf-16') as fp:
                    content = preprocessing.preprocessing(fp.read())
                    documents.append(content)
            total_files += 1
            end = time.time()
            total_time += end - begin
            if total_time > 3600:
                hours = int(total_time / 60)
                remain = total_time - hours * 3600
                minutes = int(remain / 60)
                seconds = remain - minutes * 60
                print('\rtotal files = %d - time = %dh %dm %.2fs' % (total_files, hours, minutes, seconds)),
            if total_time > 60:
                minutes = int(total_time / 60)
                seconds = total_time - minutes * 60
                print('\rtotal files = %d - time = %dm %.2fs' % (total_files, minutes, seconds)),
            else:
                print '\rtotal files = %d - time = %.2fs' % (total_files, total_time),
            sys.stdout.flush()
    print('')
    return documents