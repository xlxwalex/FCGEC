'''
Definitions of Consts
'''

# Attention Mask Const (-inf)
MASK_LOGIT_CONST = 1e9

# Tagger Map
INSERT_TAG = 'I'
MODIFY_TAG = 'MI'
KEEP_TAG = 'K'
DELETE_TAG = 'D'
MOIFY_ONLY_TAG = 'M'
MODIFY_DELETE_TAG = 'MD'
TAGGER_MAP = {'PAD' : 0, KEEP_TAG : 1, DELETE_TAG : 2, INSERT_TAG : 3, MOIFY_ONLY_TAG: 4, MODIFY_TAG : 5, MODIFY_DELETE_TAG : 6}
TAGGER_MAP_INV = ['PAD', KEEP_TAG, DELETE_TAG, INSERT_TAG, MOIFY_ONLY_TAG, MODIFY_TAG, MODIFY_DELETE_TAG]

# Generators
GEN_KEEP_LABEL = 0
MASK_SYMBOL = '[MASK]'
MASK_LM_ID = 103

# Type Map
TYPE_MAP = {'IWO' : 0, 'IWC' : 1, 'SC' : 2, 'ILL' : 3, 'CM' : 4, 'CR' : 5, 'AM' :6}
TYPE_MAP_INV = ['语序不当', '搭配不当', '结构混乱', '不合逻辑', '成分残缺', '成分赘余', '表意不明']
TYPE_MAP_INV_NEW =  {'语序不当' : 'IWO', '搭配不当' : 'IWC', '结构混乱' :'SC', '不合逻辑' : 'ILL', '成分残缺' : 'CM', '成分赘余' : 'CR', '表意不明' : 'AM'}

# Pipeline
TEMPLATE_FILE_NAME = 'app/template/fcgec-template-v2.0.docx'
CACHE_DIR = 'app/cache'
SPLIT_VOCAB = ['？', '！', '。', '\n', '；', '?', '!', ';']
INNER_VOCAB = ['，', '：', '"', '、', ',', ':', '”']
BINARY_COLOR = 'C71585'
UNDER_COLOR  = 'DC143C'
TEXT_COLOR   = '000000'