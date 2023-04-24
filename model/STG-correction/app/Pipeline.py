from argparse import Namespace
import torch
from utils import get_device, base_context, split_sentence, collate_fn_base, collate_fn_demo, TEXT_COLOR, BINARY_COLOR, UNDER_COLOR
from .demo_utils import transform2inform, context2docs
import time
from .ReportDataset import ReportDataset
from .ModelBucket import ModelBucketV1
from torch.utils.data import DataLoader
import numpy as np
from docxtpl import RichText

class Pipeline(object):
    def __init__(self, args_binary : Namespace,  args_demo : Namespace):
        self.args_binary = args_binary
        self.args_demo = args_demo
        # self.device = get_device(args_demo.cuda, args_demo.gpu_id)
        self.device = torch.device("cpu") # RECOMMAND
        self.model_bucket = ModelBucketV1(args_demo, self.device, binary=True, switch=True, taggen=True, checkpoints_name='checkpoint.pt')
        print(">> Model initialized.")

    def __call__(self, documents : str, report_name : str = None, export_path : str = None, doc_type : str= 'pdf') -> dict:
        '''
        Pipline processor for fcgec reporter
        :param documents: input docs [str]
        :return: path [str]
        '''
        context = base_context(report_name, documents, switch=True, modify=True)
        sentences = split_sentence(documents, self.args_demo.padding_size)
        data_bucket = ReportDataset(self.args_demo, sentences)
        # Binary Judge
        start_time = time.time()
        data_bucket.binary()
        # binary_results, types_results = self.binary_report(data_bucket)
        switch_results, switch_tags, switch_pointers = self.switch_report(data_bucket)
        switch_inform = (switch_results, switch_tags, switch_pointers)
        # Tagger Judge# Generate fake labels
        data_bucket.tagger(switch_results, switch_tags)
        tagger_results, generate_tags = self.tagger_process(data_bucket)
        # Generate Judge
        data_bucket.generate(generate_tags)
        generate_results, modified_text = self.generate_process(data_bucket)
        # Details Generator
        details_modification = transform2inform(data_bucket, generate_results, (tagger_results, generate_tags), switch_inform)
        detail_texts = self.details_generate(sentences, operate_reports=details_modification)
        context['details'] = self.pack_textwtype(sentences, detail_texts)
        context['process_time'] = '{:.5f}s'.format(time.time() - start_time)
        # Modified Sentence
        context['modify_text'] = self.process_modified_sentence(data_bucket, modified_text, generate_tags[0])
        path = context2docs(context, export_path, doc_type)
        return path

    def binary_report(self, data_bucket : ReportDataset) -> tuple:
        BinaryLoader = DataLoader(data_bucket, batch_size=self.args_binary.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_base)
        results, type_collect = [], []
        for _, batch_data in enumerate(BinaryLoader):
            tokens = batch_data
            result, types = self.model_bucket.binary_process(tokens)
            results.extend(result)
            type_collect.extend(types)
        return results, type_collect

    def switch_report(self, data_bucket : ReportDataset) -> tuple:
        SwitchLoader = DataLoader(data_bucket, batch_size=self.args_demo.batch_size, shuffle=False, drop_last=False,collate_fn=collate_fn_demo)
        results, sw_flag_collect, sw_preds_collection = [], [], []
        for _, batch_data in enumerate(SwitchLoader):
            tokens = batch_data
            result, sw_preds = self.model_bucket.switch_process(tokens)
            results.extend(result)
            sw_preds_collection.extend(sw_preds)
            sw_flag_collect.extend([True if np.max(np.diff(sw_preds[idx])) > 1 else False for idx in range(len(batch_data))])
        return results, sw_flag_collect, sw_preds_collection

    def tagger_process(self, data_bucket : ReportDataset) -> tuple:
        TaggerLoader = DataLoader(data_bucket, batch_size=self.args_demo.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_demo)
        results, gen_inform = [[], []], [[], []]
        for _, batch_data in enumerate(TaggerLoader):
            tokens = batch_data
            result, gen = self.model_bucket.tagger_process(tokens)
            results[0].extend(result[0])
            results[1].extend(result[1])
            gen_inform[0].extend(gen[0])
            gen_inform[1].extend(gen[1])
        return results, gen_inform

    def generate_process(self, data_bucket : ReportDataset) -> tuple:
        GenerateLoader = DataLoader(data_bucket, batch_size=self.args_demo.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_demo)
        results_token, modified_sentences = [], []
        for _, batch_data in enumerate(GenerateLoader):
            result, modified = self.model_bucket.generate_process(batch_data)
            results_token.extend(result)
            modified_sentences.extend(modified)
        return results_token, modified_sentences

    def details_generate(sself, sentences : list, binary_reports : list = None, operate_reports : list = None) -> list:
        detail_text = []
        assert  operate_reports is not None
        for index in range(len(sentences)):
            rt = RichText()
            if operate_reports[index][0] == '该句子没有语病': rt.add(operate_reports[index][0])
            else:
                op_str = ''
                for opidx in range(len(operate_reports[index])):op_str += ('[{}] {}\n'.format(opidx + 1, operate_reports[index][opidx]))
                rt.add(op_str, color=TEXT_COLOR)
            detail_text.append(rt)
        return detail_text

    def pack_textwtype(self, detail_text : list, detail_type : list) -> list:
        pack_details = []
        assert len(detail_text) == len(detail_type)
        detail_num = len(detail_text)
        for didx in range(detail_num):
            rt_text = RichText()
            texts, types = detail_text[didx], detail_type[didx]
            rt_text.add(texts, color=TEXT_COLOR)
            pack_details.append({'detail_text' : rt_text, 'detail_type' : types})
        return pack_details

    def process_modified_sentence(self, data_bucket: ReportDataset, modified_sentences: list, tagger_sentences: list):
        sentences, genidx = [], 0
        for idx in range(len(data_bucket.filter_flag)):
            flag = data_bucket.filter_flag[idx]
            if flag: sentences.append(''.join(data_bucket.tokenizer.convert_ids_to_tokens(tagger_sentences[idx][1:-1])).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '').replace('[SEP]', '"'))
            else:
                sentences.append(''.join(data_bucket.tokenizer.convert_ids_to_tokens(modified_sentences[genidx][1:-1])).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '').replace('[SEP]', '"'))
                genidx += 1
        return ''.join(sentences)