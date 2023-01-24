# Copyright 2022 The ZJU MMF Authors (Lvxiaowei Xu, Jianwang Wu, Jiawei Peng, Jiayu Fu and Ming Cai *).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Global------------------------------------------------------------
COLLOCATION = True # 主要用于Modify和Insert操作，会判断是否根据词组进行修改
# ------------------------------------------------------------------
from collections import defaultdict
from copy import deepcopy
import numpy as np
try:
    import jieba
    jieba_flag = True
    jieba.setLogLevel(jieba.logging.INFO)
except:
    if COLLOCATION: print('You need to install `jieba` first to activate collocation function.')
    jieba_flag = False
    COLLOCATION = False

def min_dist_opt(s1, s2):
    if s1 == s2:
        opt = {}
    elif is_same_group(s1, s2):
        new_idx = get_new_idx(s1, s2)
        opt = {'Switch': new_idx}
    else:
        opt = levenshtein(s1, s2)
    return opt

def is_same_group(s1: str, s2: str) -> bool:
    if len(s1) != len(s2):
        return False
    cnt = defaultdict(int)
    for w in s1:
        cnt[w] += 1
    for w in s2:
        if cnt[w] < 1:
            return False
        cnt[w] -= 1
    return True

def get_new_idx(s1: str, s2: str) -> list:
    idxs = get_common_group(s1, s2)
    if idxs:
        return idxs
    new_idx = [0] * len(s1)
    for i in range(len(s1)):
        new_idx[i] = i
    same_pos = get_same_pos(s1, s2)
    word2idx = defaultdict(list)
    for i, w in enumerate(s1):
        if not same_pos[i]:
            word2idx[w].append(i)
    for i, w in enumerate(s2):
        if not same_pos[i]:
            new_idx[i] = word2idx[w][0]
            word2idx[w].pop(0)
    return new_idx

def get_same_pos(s1: str, s2: str) -> list:
    same_pos = [0] * len(s1)
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            same_pos[i] = 1
    return same_pos

def switch_swap(sen: str, g1: list, g2: list) -> list:
    if g1[0] > g2[0]:
        g1, g2 = g2, g1
    origin = list(range(len(sen)))
    if is_punct(sen[g1[0]]):
        g1[0] += 1
    if is_punct(sen[g1[1] - 1]):
        g1[1] -= 1
    if g2[0] < len(sen) and is_punct(sen[g2[0]]):
        g2[0] += 1
    if g2[1] <= len(sen) and is_punct(sen[g2[1] - 1]):
        g2[1] -= 1
    res = origin[:g1[0]] + list(range(g2[0], g2[1])) + \
        origin[g1[1]:g2[0]] + list(range(g1[0], g1[1])) + origin[g2[1]:]
    return res

def get_switch_result(sen: str, idxs: list) -> str:
    res = ''.join([sen[i] for i in idxs])
    return res

def is_punct(c):
    punct = ',.:;，。：；、．\'\"‘’“”…'
    if c in punct:
        return True
    return False

def get_common_group(s1, s2):
    dp = [[0] * len(s2) for _ in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i > 0 and j > 0:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = 1
    dp = np.array(dp).max(axis=1)
    maxl = dp.max()
    if maxl < 2:
        return False
    g1 = np.where(dp == maxl)[0]
    if g1.size == 1:
        i, j = g1[0] - maxl + 1, g1[0] + 1
        dp[i:j] = 0
        maxl = dp.max()
        g2 = np.where(dp == dp.max())[0]
        for k in range(len(g2)):
            i_, j_ = g2[k] - maxl + 1, g2[k] + 1
            idxs = switch_swap(s1, [i, j], [i_, j_])
            s3 = get_switch_result(s1, idxs)
            if s3 == s2:
                return idxs
    else:
        for p in range(len(g1)):
            i, j = g1[p] - maxl + 1, g1[p] + 1
            for q in range(p+1, len(g1)):
                i_, j_ = g1[q] - maxl + 1, g1[q] + 1
                idxs = switch_swap(s1, [i, j], [i_, j_])
                s3 = get_switch_result(s1, idxs)
                if s3 == s2:
                    return idxs
    return False

def levenshtein(s1, s2):
    dp = [[0] * (len(s2) + 1) for i in range(len(s1) + 1)]
    path = [[0] * (len(s2) + 1) for i in range(len(s1) + 1)]
    lastop = {
        0: (-1, 0),
        1: (0, -1),
        2: (-1, -1),
        3: (-1, -1)
    }
    for j in range(len(dp[0])):
        dp[0][j] = j
        path[0][j] = 2
    for i in range(len(dp)):
        dp[i][0] = i
        path[i][0] = 1
    for i in range(1, len(dp)):
        for j in range(1, len(dp[i])):
            c1 = dp[i-1][j] + 1
            c2 = dp[i][j-1] + 1
            c3 = dp[i-1][j-1] + 1
            if s1[i-1] == s2[j-1]:
                c3 -= 1
                path[i][j] += 8
            dp[i][j] = min(c1, c2, c3)
            if dp[i][j] == c1:
                path[i][j] += 1
            if dp[i][j] == c2:
                path[i][j] += 2
            if dp[i][j] == dp[i - 1][j - 1] + 1:
                path[i][j] += 4
    ops = []
    get_ops(path, len(s1), len(s2), lastop, ops, [])
    ops = [parse_ops(s1, s2, op)[1] for op in ops]
    great_ops = ops[0]
    opcount = lambda op: len(op)
    oplen = lambda op: sum(len(p) for p in op.values())
    for op in ops[1:]:
        if opcount(op) <= opcount(great_ops) and oplen(op) < oplen(great_ops):
            great_ops = op
    simplify_ops(great_ops)
    return dict(great_ops)


def get_ops(path, i, j, lastop, res, temp):
    if i == 0 and j == 0:
        res.append([t for t in temp[::-1]])
        return
    for op in range(4):
        if (path[i][j] >> op) & 1:
            temp.append((i, j, op))
            get_ops(path, i+lastop[op][0], j+lastop[op][1], lastop, res, temp)
            temp.pop()
            break

def parse_ops(s1, s2, ops):
    tag = {
        0: "Delete",
        1: "Insert",
        2: "Modify",
        3: "Copy"
    }
    res = defaultdict(list)
    for op in ops:
        if op[2] == 0:
            res[tag[0]].append((op[0] - 1, s1[op[0]-1]))
        elif op[2] == 1:
            res[tag[1]].append((op[0] - 1, s2[op[1]-1]))
        elif op[2] == 2:
            res[tag[2]].append((op[0] - 1, s1[op[0] - 1], s2[op[1] - 1]))
    ret = defaultdict(list)
    if res.get("Insert"):
        idxs, words = zip(*res["Insert"])
        idxs_, words_ = [idxs[0]], [words[0]]
        for i in range(1, len(idxs)):
            if idxs[i] != idxs[i-1]:
                idxs_.append(idxs[i])
                words_.append(words[i])
            else:
                words_[-1] += words[i]
        temp = {}
        for i, idx in enumerate(idxs_):
            temp["pos"] = idx
            temp["tag"] = "INS_" + str(len(words_[i]))
            temp["label"] = words_[i]
            ret["Insert"].append(deepcopy(temp))

    if res.get("Delete"):
        idxs, words = zip(*res["Delete"])
        temp = {}
        temp["pos"] = idxs
        temp["label"] = ''.join(words)
        ret["Delete"].extend(idxs)
    if res.get("Modify"):
        idxs, words, new_words = zip(*res["Modify"])
        idxs_, words_, new_words_ = [idxs[0]], [words[0]], [new_words[0]]
        for i in range(1, len(idxs)):
            if idxs[i] != idxs[i-1] + 1:
                idxs_.append(idxs[i])
                words_.append(words[i])
                new_words_.append(new_words[i])
            else:
                words_[-1] += words[i]
                new_words_[-1] += new_words[i]
        temp = {}
        for i, idx in enumerate(idxs_):
            words = words_[i]
            new_words = new_words_[i]
            if COLLOCATION:
                idx, words, new_words  = post_collocation(s1, idx, words, new_words)
            temp["pos"] = idx
            temp["tag"] = "MOD_" + str(len(words))
            temp["label"] = new_words
            ret["Modify"].append(deepcopy(temp))
    return res, ret

def post_collocation(s1, index, words, new_words):
    s_cut = jieba.lcut(s1)
    init, mapper = 0, {}
    for i, element in enumerate(s_cut):
        for li in range(len(element)):
            mapper[init+li] = init
        init += len(element)
    common_str = s1[mapper[index]:index]
    return mapper[index], common_str + words, common_str +new_words

def simplify_ops(ops):
    ret = ops
    if ret.get('Modify') and ret.get('Insert'):
        ins = ret['Insert']
        mod = ret['Modify']
        for i in range(len(mod)):
            idx = mod[i]['pos'] + len(mod[i]['label'])
            for j in range(len(ins)-1, -1, -1):
                if ins[j]['pos'] + 1 == idx:
                    mod[i]['tag'] += ''.join(['+', ins[j]['tag']])
                    mod[i]['label'] += ins[j]['label']
                    ins.pop(j)
        if len(ins) == 0:
            ret.pop('Insert')
    if ret.get('Modify') and ret.get('Delete'):
        dels = ret['Delete']
        mod = ret['Modify']
        for i in range(len(mod)):
            idx = mod[i]['pos'] + len(mod[i]['label'])
            k = 0
            while idx in dels:
                dels.remove(idx)
                idx += 1
                k += 1
            if k > 0:
                mod[i]['tag'] = 'MOD_{}+DEL_{}'.format(k + len(mod[i]['label']), k)
        if len(dels) == 0:
            ret.pop('Delete')


def selectMinOpt(s1, s2s):
    if len(s2s) < 1:
        return {}
    opts = [(i, min_dist_opt(s1, s2)) for i, s2 in enumerate(s2s)]
    opts.sort(key=lambda x: len(x[1]))
    return opts[0] if len(opts[0][1]) != 0 or len(opts) == 1 else opts[1]

def clean(s):
    return s.replace(' ', '').replace(',', '，').replace('.', '。')

if __name__ == '__main__':
    # Examples
    # - INSERT
    origin = "培养学生的思维能力，是衡量一节课是否成功的重要标准。"
    correction = "能否培养学生的思维能力，是衡量一节课是否成功的重要标准。"
    opt = min_dist_opt(origin, correction)
    print("INSERT example: {}".format(opt))

    # - DELETE
    origin = "石济高铁正式开通，两地旅行时间从原来最快约四个小时缩短到约两个小时左右，标志着我国“四纵四横”高铁网中的“四横”完美收官。"
    correction = "石济高铁正式开通，两地旅行时间从原来最快约四个小时缩短到约两个小时，标志着我国“四纵四横”高铁网中的“四横”完美收官。"
    opt = min_dist_opt(origin, correction)
    print("DELETE example: {}".format(opt))

    # - MODIFY
    origin = "随着可操纵粒子数的增加，量子计算机计算能力呈指数增长，可以为经典计算机无法解决的大规模计算难题提取有效解决方案。"
    correction = "随着可操纵粒子数的增加，量子计算机计算能力呈指数增长，可以为经典计算机无法解决的大规模计算难题提供有效解决方案。"
    opt = min_dist_opt(origin, correction)
    print("MODIFY example: {}".format(opt))

    # - SWITCH
    origin = "军工企业与地方企业合作，发挥各自优势，共同生产和研制了高质量的民用产品。"
    correction = "军工企业与地方企业合作，发挥各自优势，共同研制和生产了高质量的民用产品。"
    opt = min_dist_opt(origin, correction)
    print("SWITCH example: {}".format(opt))
