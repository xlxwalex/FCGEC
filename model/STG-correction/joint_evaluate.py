import os
import warnings
warnings.filterwarnings("ignore")
from config import evaljoint_parse
# Import torch
from torch.utils.data import DataLoader
import torch
# Import Dataset
from DataProcessor import TaggerDataset as TaggerDataset, SwitchDataset, GeneratorDataset
# Import utils
from utils.collate import collate_fn_base as collate_fn_base, collate_fn_tagger_V2 as collate_fn_tagger
from utils import get_device, TAGGER_MAP, report_pipeline_output
from Model import  JointModelwithEncoder as JointModel
from utils import padding, attention_mask, SwitchSearch, convert_spmap_tg, convert_spmap_sw
from tqdm import tqdm
from utils import reconstruct_tagger_V2 as reconstruct_tagger, fillin_tokens, convert_spmap2tokens
import numpy as np
import scipy.io as sio

def _apply_switch_operator(wd_idxs: list, switch_ops: list) -> list:
    res = []
    for lidx in range(len(wd_idxs)):
        post_token = [101]
        switch_pred = switch_ops[lidx]
        sw_pidx = switch_pred[0]
        wd_idx = wd_idxs[lidx]
        while sw_pidx not in [0, -1]:
            post_token.append(wd_idx[sw_pidx])
            sw_pidx = switch_pred[sw_pidx]
            if wd_idx[sw_pidx] == 102: switch_pred[sw_pidx] = 0
        res.append(post_token)
    return res

def evaluate(args):
    device = get_device(args.cuda, args.gpu_id)
    # Switch
    test_dir = os.path.join(args.data_base_dir, 'test.csv')
    if os.path.exists(os.path.join(args.data_base_dir, 'uuid.mat')):
        uuid = sio.loadmat(os.path.join(args.data_base_dir, 'uuid.mat'))['uuid']
    else:
        uuid = None

    switch_test = SwitchDataset(args, test_dir, 'test')
    params = torch.load(os.path.join(args.checkpoints, args.checkp, 'checkpoint.pt'))["model"]
    TestLoader = DataLoader(switch_test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_base)
    decoder = SwitchSearch(args, args.sw_mode)
    model = JointModel(args, device)
    model.load_state_dict(params)
    model = model.to(device)
    model.eval()

    pred_logits, truth_label, met_masks, tokens_ls, switch_preds, switch_gts, switch_tokens = None, [], [], [], [], [], []
    for step, batch_data in enumerate(tqdm(TestLoader, desc='Processing')):
        # Process Data
        tokens, labels = batch_data
        tokens_ls.extend(tokens)
        padded = padding(tokens, args.padding_size, args.padding_val)
        attn_mask = attention_mask(padded, args.padding_val).to(device)
        padded = torch.from_numpy(padded).to(device)
        # Model Value
        with torch.no_grad():
            pointer_logits = model.switch(padded, attn_mask)
        truths = padding(labels, args.padding_size,args.padding_val)
        pred_logits = torch.cat((pred_logits, pointer_logits), dim=0) if pred_logits is not None else pointer_logits
        met_masks.extend(attn_mask.detach().cpu().numpy())
        truth_label.extend(truths)
    pred_label    = decoder(pred_logits.detach().cpu(), met_masks)
    switch_tokens = _apply_switch_operator(tokens_ls, pred_label)
    switch_gts    = _apply_switch_operator(tokens_ls, truth_label)

    # Special map
    if hasattr(args, 'sp_map') and args.sp_map:
        sp_maps = [pt.spmap for pt in switch_test.point_seq]
        sw_spmaps = convert_spmap_sw(sp_maps, pred_label)
    else: sw_spmaps = None

    print('Construct Tagger Data')
    tagger_test = TaggerDataset(args, test_dir, 'test', switch_tokens)
    TestLoader = DataLoader(tagger_test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_tagger)
    tagger_gts_dataset = TaggerDataset(args, test_dir, 'trains')  # EM
    gts_tagger, gts_comb = padding(tagger_gts_dataset.tagger_idx, args.padding_size, args.padding_val), padding(tagger_gts_dataset.comb_label, args.padding_size, args.padding_val)  # EM
    tag_construct_gts = (gts_tagger, gts_comb)  # EM

    pred_tagger, pred_comb, met_masks, tagger_tokens = [], [], [], []
    for step, batch_data in enumerate(tqdm(TestLoader, desc='Processing Tagger')):
        # Process Data
        tokens = batch_data
        padded_token = padding(tokens, args.padding_size, args.padding_val)
        tagger_tokens.extend(padded_token.tolist())
        attn_mask = attention_mask(padded_token, args.padding_val).to(device)
        token_padded = torch.from_numpy(padded_token).to(device)
        # Model Value
        with torch.no_grad():
            tagger_logits, comb_logits = model.tagger(token_padded, attn_mask)
        tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
        comb_preds = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
        pred_tagger.extend(tagger_preds)
        pred_comb.extend(comb_preds)
        met_masks.extend(attn_mask.detach().cpu().numpy())

    print('Construct Generator Data')
    tag_construct = (pred_tagger, pred_comb)
    tag_tokens, mlm_tgt_masks, tg_mapper = reconstruct_tagger(np.array(tagger_tokens), tag_construct)
    tag_gts_tokens, _, _ = reconstruct_tagger(padding(switch_gts, args.padding_size, args.padding_val),  tag_construct_gts)  # EM
    generator_test = GeneratorDataset(args, test_dir, 'test', tag_tokens, mlm_tgt_masks)
    TestLoader = DataLoader(generator_test, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_base)

    # Special map
    if hasattr(args, 'sp_map') and args.sp_map:
        tg_spmaps = convert_spmap_tg(sw_spmaps, tg_mapper)
    else:
        tg_spmaps = None

    pred_mlm, truth_mlm, met_masks = [], [], []
    for step, batch_data in enumerate(tqdm(TestLoader, desc='Processing Generator')):
        # Process Data
        tokens, label = batch_data
        padded_gens = padding(tokens, args.padding_size, args.padding_val)
        gen_attn_mask = attention_mask(padded_gens, args.padding_val).to(device)
        gen_token_tensor = torch.from_numpy(padded_gens).to(device)
        padded_mlm_tgt_mask = padding(label, args.padding_size, args.padding_val)
        tgt_mlm_tensor = torch.from_numpy(padded_mlm_tgt_mask).to(device)
        # Model Value
        with torch.no_grad():
            mlm_logits, tgt_mlm, _ = model.generator(gen_token_tensor, tgt_mlm_tensor, gen_attn_mask)
        # Preds
        token_preds = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
        token_truth = tgt_mlm.detach().cpu().numpy()
        pred_mlm.extend(token_preds)
        truth_mlm.extend(token_truth)
        met_masks.extend(gen_attn_mask.detach().cpu().numpy())

    print('>>> Start to constrcut final output')
    outputs = fillin_tokens(tag_tokens, mlm_tgt_masks, pred_mlm)
    switch_tokens = [''.join(convert_spmap2tokens(switch_test.tokenizer.convert_ids_to_tokens(ele[1:-1]), sw_spmaps[i])).replace('##', '').replace('[PAD]', '').replace('[UNK]', '') for i, ele in enumerate(switch_tokens)] \
        if hasattr(args, 'sp_map') and args.sp_map else  [''.join(switch_test.tokenizer.convert_ids_to_tokens(ele[1:-1])).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '') for ele in switch_tokens]
    tagger_tokens = [''.join(convert_spmap2tokens(switch_test.tokenizer.convert_ids_to_tokens(ele[1:-1]), tg_spmaps[i])).replace('##', '').replace('[PAD]', '').replace('[UNK]', '') for i, ele in enumerate(tag_tokens)] \
        if hasattr(args, 'sp_map') and  args.sp_map else [''.join(tagger_test.tokenizer.convert_ids_to_tokens(ele[1:-1])).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '') for ele in tag_tokens]
    generate_tokens = [''.join(convert_spmap2tokens(switch_test.tokenizer.convert_ids_to_tokens(ele[1:-1]), tg_spmaps[i])).replace('##', '').replace('[PAD]', '').replace('[UNK]', '') for i, ele in enumerate(outputs)] \
        if hasattr(args, 'sp_map') and  args.sp_map else [''.join(generator_test.tokenizer.convert_ids_to_tokens(ele[1:-1])).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '') for ele in outputs]
    report_pipeline_output(os.path.join(args.data_base_dir, args.export), switch_test.sentences, switch_test.label, switch_tokens, tagger_tokens, generate_tokens, uuid=uuid)
    print('Final output saved at %s' % os.path.join(args.data_base_dir, args.export))

if __name__ == '__main__':
    args = evaljoint_parse()
    # Process Args
    args.tagger_classes = len(TAGGER_MAP.keys())
    evaluate(args)
