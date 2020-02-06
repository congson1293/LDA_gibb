# -*- encoding: utf-8 -*-

import unicodedata
import regex
from nlp_tools import tokenizer
try:
    from pyvi.pyvi import ViPosTagger
except:
    from pyvi import ViPosTagger


importance_pos = {'N':True, 'Np':True, 'Ny':True, 'Nb' : True,
                  'V':True, 'Vb':True, 'Vy':True}

my_regex = regex.regex()


def is_exist(postag):
    try:
        _ = importance_pos[postag]
        return True
    except:
        return False


def remove_stop_postag(data):
    content = map(lambda x: ViPosTagger.postagging(x),
                  data.split(u'\n'))
    clean_content = []
    for info in content:
        sen = []
        for i in xrange(len(info[0])):
            if is_exist(info[1][i]):
                sen.append(info[0][i])
        clean_content.append(u' '.join(sen))
    return u'\n'.join(clean_content)


def preprocessing(data, tokenize=True):
    try:
        data = unicodedata.normalize('NFKC', data)
    except:
        data = unicode(data)
    if tokenize:
        data = tokenizer.predict(data)
    data = my_regex.detect_url.sub(u'', data)
    data = my_regex.detect_url2.sub(u'', data)
    data = my_regex.detect_email.sub(u'', data)
    data = my_regex.detect_datetime.sub(u'', data)
    data = my_regex.detect_num.sub(u'', data)
    data = my_regex.normalize_special_mark.sub(u' \g<special_mark> ', data)
    data = my_regex.detect_exception_chars.sub(u'', data)
    data = my_regex.detect_special_mark.sub(u'', data)
    data = my_regex.detect_special_mark2.sub(u'', data)
    data = my_regex.detect_special_mark3.sub(u'', data)
    data = my_regex.normalize_space.sub(u' ', data)

    data = remove_stop_postag(data)
    return data.strip()