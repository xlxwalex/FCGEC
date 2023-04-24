import os
import warnings
warnings.filterwarnings("ignore")
from config import evaljoint_parse
import torch
from utils import get_device, TAGGER_MAP
from Model import  JointModel
from transformers import BertTokenizer
from utils import padding, attention_mask, SwitchSearch
from utils import reconstruct_tagger_V2 as reconstruct_tagger, fillin_tokens, _apply_switch_operator
import numpy as np

# Args
args = evaljoint_parse()
args.tagger_classes = len(TAGGER_MAP.keys())
device = get_device(args.cuda, args.gpu_id)
# Model
params = torch.load(os.path.join(args.checkpoints, args.checkp, 'checkpoint.pt'), map_location='cpu')["model"]
model = JointModel(args, device)
model.load_state_dict(params)
model = model.to(device)
model.eval()
# Components
tokenizer = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
decoder = SwitchSearch(args, args.sw_mode)
print('>> Model initialized.')

def parse_tokens(tokens: list):
    return ''.join(tokens[1:-1]).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '').replace('[SEP]', '')

def single_sentence_corrector(sentence: str):
    tokens = [tokenizer.convert_tokens_to_ids(['[CLS'] + tokenizer.tokenize(sentence) + ['[SEP]'])]
    # Switch
    padded = padding(tokens, args.padding_size, args.padding_val)
    attn_mask = attention_mask(padded, args.padding_val).to(device)
    padded = torch.from_numpy(padded).to(device)
    with torch.no_grad(): pointer_logits = model.switch(padded, attn_mask)
    pred_label = decoder(pointer_logits.detach().cpu(), attn_mask.detach().cpu())
    switch_tokens = _apply_switch_operator(tokens, pred_label)
    # Tagger
    padded = padding(switch_tokens, args.padding_size, args.padding_val)
    attn_mask = attention_mask(padded, args.padding_val).to(device)
    padded = torch.from_numpy(padded).to(device)
    with torch.no_grad(): tagger_logits, comb_logits = model.tagger(padded, attn_mask)
    tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
    comb_preds = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
    tag_construct = (tagger_preds, comb_preds)
    tagger_tokens_ids, mlm_tgt_masks, _ = reconstruct_tagger(padded.detach().cpu().numpy(), tag_construct)
    # Generator
    tagger_tokens = tokenizer.convert_ids_to_tokens(tagger_tokens_ids[0])
    if max(mlm_tgt_masks[0]) < 1: return parse_tokens(tagger_tokens)  # direct return (do not use generator)
    padded = padding(tagger_tokens_ids, args.padding_size, args.padding_val)
    attn_mask = attention_mask(padded, args.padding_val).to(device)
    padded = torch.from_numpy(padded).to(device)
    padded_mask = torch.from_numpy(padding(mlm_tgt_masks, args.padding_size, args.padding_val)).to(device)
    with torch.no_grad(): mlm_logits, tgt_mlm, _ = model.generator(padded, padded_mask, attn_mask)
    token_preds = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
    generate_token_ids = fillin_tokens(tagger_tokens_ids, mlm_tgt_masks, token_preds)
    return parse_tokens(tokenizer.convert_ids_to_tokens(generate_token_ids[0]))


if __name__ == '__main__':
    while True:
        sentence = input('Input the incorrect sentence (q for quit):')
        if sentence == 'q': break
        print('>> corrected sentence: {}'.format(single_sentence_corrector(sentence)))
    print('Quit.')