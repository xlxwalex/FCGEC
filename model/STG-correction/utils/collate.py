# Collate_fn of DataLoader(BERTDataset / SwitchDataset / GeneratorDataset)
def collate_fn_base(batch):
    dim = len(batch[0].keys())
    if dim == 2:  # Train DataLoader
        tokens    = [item['token'] for item in batch]
        labels = [item['label'] for item in batch]
        return (tokens, labels)
    elif dim == 1: # Test DataLoader
        tokens    = [item['token'] for item in batch]
        return (tokens)
    else:
        raise Exception('Error Batch Input, Please Check.')

def collate_fn_bertbase_tti(batch):
    dim = len(batch[0].keys())
    if dim == 3:  # Train DataLoader
        tokens    = [item['token'] for item in batch]
        types     = [item['type'] for item in batch]
        labels = [item['label'] for item in batch]
        return (tokens, types, labels)
    elif dim == 2: # Test DataLoader
        tokens    = [item['token'] for item in batch]
        types = [item['type'] for item in batch]
        return (tokens, types)
    else:
        raise Exception('Error Batch Input, Please Check.')

# Collate_fn of DataLoader(TaggerDdataset)
def collate_fn_tagger(batch):
    dim = len(batch[0].keys())
    if dim == 4:  # Train DataLoader
        tokens = [item['token'] for item in batch]
        tagger = [item['tagger'] for item in batch]
        ins    = [item['ins'] for item in batch]
        mod    = [item['mod'] for item in batch]
        return (tokens, tagger, ins, mod)
    elif dim == 1: # Test DataLoader
        tokens = [item['token'] for item in batch]
        return (tokens)
    else:
        raise Exception('Error Batch Input, Please Check.')

# Collate_fn of DataLoader(TaggerDdatasetV2)
def collate_fn_tagger_V2(batch):
    dim = len(batch[0].keys())
    if dim == 3:  # Train DataLoader
        tokens = [item['token'] for item in batch]
        tagger = [item['tagger'] for item in batch]
        comb   = [item['comb'] for item in batch]
        return (tokens, tagger, comb)
    elif dim == 1: # Test DataLoader
        tokens = [item['token'] for item in batch]
        return (tokens)
    else:
        raise Exception('Error Batch Input, Please Check.')

# Collate_fn of DataLoader(TaggerDdatasetV2TTI)
def collate_fn_tagger_V2TTI(batch):
    dim = len(batch[0].keys())
    if dim == 4:  # Train DataLoader
        tokens = [item['token'] for item in batch]
        tagger = [item['tagger'] for item in batch]
        comb   = [item['comb'] for item in batch]
        type   = [item['type'] for item in batch]
        return (tokens, tagger, comb, type)
    elif dim == 2: # Test DataLoader
        tokens = [item['token'] for item in batch]
        type = [item['type'] for item in batch]
        return (tokens, type)
    else:
        raise Exception('Error Batch Input, Please Check.')

# Collate_fn of DataLoader(JointDataset)
def collate_fn_joint(batch):
    dim = len(batch[0].keys())
    if dim == 10:  # Train DataLoader
        wid_ori   = [item['wid_ori'] for item in batch]
        wid_tag   = [item['wid_tag'] for item in batch]
        wid_gen   = [item['wid_gen'] for item in batch]

        tag_label = [item['tag_label'] for item in batch]
        ins_label = [item['ins_label'] for item in batch]
        mod_label = [item['mod_label'] for item in batch]

        bi_label  = [item['bilabel'] for item in batch]
        typelabel = [item['typelabel'] for item in batch]
        sw_label  = [item['swlabel'] for item in batch]
        mlmlabel  = [item['mlmlabel'] for item in batch]

        wid_collection  = (wid_ori, wid_tag, wid_gen)
        tag_collection  = (tag_label, ins_label, mod_label)
        spec_collection = (bi_label, typelabel, sw_label, mlmlabel)

        return wid_collection, tag_collection, spec_collection
    else:
        raise Exception('Error Batch Input, Please Check.')

# Collate_fn of DataLoader(JointDataset)
def collate_fn_jointV2(batch):
    dim = len(batch[0].keys())
    if dim == 7:  # Train DataLoader
        wid_ori   = [item['wid_ori'] for item in batch]
        wid_tag   = [item['wid_tag'] for item in batch]
        wid_gen   = [item['wid_gen'] for item in batch]

        tag_label = [item['tag_label'] for item in batch]
        comb_label = [item['comb_label'] for item in batch]

        sw_label  = [item['swlabel'] for item in batch]
        mlmlabel  = [item['mlmlabel'] for item in batch]

        wid_collection  = (wid_ori, wid_tag, wid_gen)
        tag_collection  = (tag_label, comb_label)
        spec_collection = (sw_label, mlmlabel)

        return wid_collection, tag_collection, spec_collection
    else:
        raise Exception('Error Batch Input, Please Check.')


# Collate_fn of DataLoader(JointDatasetTTI)
def collate_fn_jointTTI(batch):
    dim = len(batch[0].keys())
    if dim == 8:  # Train DataLoader
        wid_ori   = [item['wid_ori'] for item in batch]
        wid_tag   = [item['wid_tag'] for item in batch]
        wid_gen   = [item['wid_gen'] for item in batch]

        tag_label = [item['tag_label'] for item in batch]
        comb_label = [item['comb_label'] for item in batch]

        type_label = [item['type'] for item in batch]
        sw_label  = [item['swlabel'] for item in batch]
        mlmlabel  = [item['mlmlabel'] for item in batch]

        wid_collection  = (wid_ori, wid_tag, wid_gen)
        tag_collection  = (tag_label, comb_label)
        spec_collection = (sw_label, mlmlabel)

        return wid_collection, tag_collection, spec_collection, type_label
    else:
        raise Exception('Error Batch Input, Please Check.')