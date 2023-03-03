INER_PUNCT = {'''“''' : '''"''', '''”“''' : '''"''', '''‘''' : "'", '''’''' : "'", '：' : ':',
              '''”''' : '''"''', '℅' : '%'}
import re
ENG_PATTERN = re.compile(r'[A-Za-z]', re.S)


class TextWash(object):
    @staticmethod
    def punc_wash(sentence : str, puncls = None, lower = True):
        if puncls is None:
            punc_ls = INER_PUNCT
        else:
            punc_ls = puncls
        if lower:
            sentence = sentence.lower()
        for ele in punc_ls:
            sentence = sentence.replace(ele, punc_ls[ele])
        return sentence

    @staticmethod
    def punc_wash_res(sentence : str, puncls = None, lower = True):
        map_special_element, post_sentence = {}, ''
        if puncls is None:
            punc_ls = INER_PUNCT
        else:
            punc_ls = puncls

        for i, item in enumerate(sentence):
            if (ord(item) > 96 and ord(item) < 123) or (ord(item) > 64 and ord(item) < 91):
                map_special_element[i] = item
                if lower:
                    post_sentence += item.lower()
                else: post_sentence += item
            elif item in punc_ls:
                post_sentence += punc_ls[item]
                map_special_element[i] = item
            else:
                post_sentence += item
        return post_sentence, map_special_element