import numpy as np
from argparse import Namespace
import torch
from utils import norm_logits, softmax_logits, padding
from copy import copy


class SwitchSearch(object):
    def __init__(self, args : Namespace, mode : str = 'vabm', pad : bool = True):
        self.args = args
        self.beamw = args.beam_width
        self.mode = mode
        self.max_length = args.padding_size
        self.use_lm = args.use_lm
        self.pad = pad
        self.padval = args.padding_val

    def __call__(self, logits : torch.Tensor, masks : list = None) -> np.array:
        if self.mode == 'vags':
            # Vanilla Greedy Search
            return self._vanilla_gready(logits)
        elif self.mode == 'rsgs':
            # Restricted Greedy Search
            return self._restrict_greedy(logits, masks)
        elif self.mode == 'rsbm':
            # Restricted Beam Search
            seqlen = logits.shape[1]
            endids = self._get_sepend_id(masks)
            sepids = [[id] for id in endids]
            legals = [list(range(id + 1)) for id in endids]
            seq_result = self._res_beamsearch_batch(logits, endids, sepids, legals)
            if self.pad:
                seq_result = padding(seq_result, seqlen, self.padval)
            return seq_result

    def _get_sepend_id(self, mask : list) -> list:
        sep_idxs = []
        if self.use_lm:
            sep_idxs = [np.where(mk == 0)[0][0] - 1 if mk[-2] != 1 else len(mk) - 2 if mk[-1] != 1 else len(mk) - 1 for mk in mask]
        return sep_idxs

    def _restrict_greedy(self, logits : torch.Tensor, masks : list) -> np.array:
        endids = self._get_sepend_id(masks)
        sequence = [self._rs_greedy_single(logits[sid], endids[sid]) for sid in range(logits.shape[0])]
        return padding(sequence, self.args.padding_size, self.padval)

    def _rs_greedy_single(self, logit: torch.Tensor, endid : int) -> list:
        seq = []
        logit = logit.numpy()
        logit[0, :endid+1] = -float("inf")
        for sid in range(endid + 1):
            if len(seq) > 0 : logit[np.array(seq), sid] = -float("inf")
            if sid < endid - 1: logit[endid, sid] = -float("inf")
            argmax_id = np.argmax(logit[:, sid])
            seq.append(argmax_id)
        assert len(seq) == endid + 1
        return seq

    def _rs_greedy_single_v2(self, logit: torch.Tensor, endid : int) -> list:
        seq, seq_map = [], {}
        back_logit = copy(logit).numpy()
        #logit = softmax_logits(logit)
        logit[0, :endid+1] = 0
        pidx, sid = 0, 0
        while sid <= endid:
            if len(seq) > 0: logit[np.array(seq), pidx] = 0
            if sid < endid - 1: logit[endid, pidx] = 0
            argmax_id = np.argmax(softmax_logits(logit[:, pidx], dim=0).numpy())
            if pidx == 0 and argmax_id == 0: argmax_id = 1
            seq.append(argmax_id)
            seq_map[pidx] = argmax_id
            pidx = argmax_id
            sid += 1
        post_seq =[]
        for i in range(endid + 1):
            if i not in seq_map.keys():
                post_seq.append(endid)
            else:
                post_seq.append(seq_map[i])
        return post_seq

    def _rs_greedy_singleV3(self, logit: torch.Tensor, endid : int) -> list:
        seq = []
        logit = logit.numpy()
        logit[0, :endid+1] = 0
        for sid in range(endid + 1):
            if len(seq) > 0 : logit[np.array(seq), sid] = 0
            if sid < endid - 1: logit[endid, sid] = 0
            argmax_id = np.argmax(logit[:, sid])
            seq.append(argmax_id)
        assert len(seq) == endid + 1
        return seq

    def _vanilla_gready(self, logits :torch.Tensor) -> np.array:
        return np.argmax(logits.numpy(), axis=1).astype('int32')

    def _res_beamsearch_batch(self, logits : torch.Tensor, endid : list, sepids : list, legals : list):
        tensor_shape = logits.shape
        logits = logits.permute(0, 2, 1)
        if len(tensor_shape) < 3:
            if isinstance(endid, list):
                endid = endid[0]
            return self._res_beamsearch_single(logits, endid, sepids, legals)
        else:
            beam_seqs = []
            batch_size = tensor_shape[0]
            for index in range(batch_size):
                logit = logits[index]
                endidx = endid[index]
                sepid = sepids[index]
                legal = legals[index]
                beam_seqs.append(self._res_beamsearch_single(logit, endidx, sepid, legal))
            return beam_seqs

    def _res_beamsearch_single(self, logits : torch.Tensor, endid : int, sepids : list, legals : list) -> list:
        beamw = copy(self.beamw)
        predicted_points =  -1 * softmax_logits(logits/2)
        sequences = [[0]]
        scores = [0]
        finished_sequences = []
        finished_scores = []
        for _ in range(self.max_length):
            assert len(sequences) == len(scores)
            candidate_scores = []
            candidate_sequences_reconstructor = []
            for j, (sequence, score) in enumerate(zip(sequences, scores)):
                sequence_set = set(sequence)
                next_scores = predicted_points[sequence[-1]]
                for index in range(endid + 1):
                    if index in sequence_set:
                        continue
                    if index not in legals:
                        continue
                    if len(sequence) == len(legals) - 1:
                        if index not in sepids:
                            continue
                    elif index in sepids and len(sepids) == 1:
                        continue

                    candidate_scores.append(score + next_scores[index])
                    candidate_sequences_reconstructor.append((j, index))

            if not candidate_scores:
                break

            if beamw < 1:
                break
            if beamw >= len(candidate_scores):
                top_n_indexes = list(range(len(candidate_scores)))
            else:
                top_n_indexes = np.argpartition(candidate_scores, beamw)[:beamw]

            new_sequences = []
            new_scores = []

            for top_n_index in top_n_indexes:
                sequence_index, token_index = candidate_sequences_reconstructor[
                    top_n_index]
                new_sequence = sequences[sequence_index] + [token_index]
                new_score = candidate_scores[top_n_index]
                if len(new_sequence) == len(legals):
                    finished_sequences.append(new_sequence)
                    finished_scores.append(-1 * new_score / len(new_sequence))
                    beamw -= 1
                else:
                    new_sequences.append(new_sequence)
                    new_scores.append(new_score)

            sequences = new_sequences
            scores = new_scores
            if beamw < 1:
                break
        if not finished_sequences:
            return None

        return finished_sequences[np.argmax(finished_scores)][1:]


