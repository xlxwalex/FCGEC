INER_PUNCT = {'''“''' : '''"''', '''”“''' : '''"''', '''‘''' : "'", '''’''' : "'", '：' : ':',
              '''”''' : '''"''', '℅' : '%'}

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