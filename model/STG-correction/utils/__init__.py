from utils.misc import _get_device as get_device
from utils.misc import set_seed
from utils.model_op import _attention_mask as attention_mask
from utils.model_op import _padding as padding, _normalize_logits as norm_logits, _softmax_logits as softmax_logits, _clip_max_generate  as clip_maxgenerate
from utils.model_op import _generate_tagger_loss_weights as tagger_loss_weights
from utils.model_op import _save_model as save_model
from utils.textwash import TextWash
from utils.mask import SelfAttentionMask, logits_mask
from utils.metric import SwitchMetric, TaggerMetric, GeneratorMetric, TaggerMetricV2
from utils.search import SwitchSearch
from utils.defines import TAGGER_MAP, TAGGER_MAP_INV,INSERT_TAG, MODIFY_TAG, KEEP_TAG, MOIFY_ONLY_TAG, DELETE_TAG, MODIFY_DELETE_TAG, MASK_SYMBOL, TYPE_MAP, TYPE_MAP_INV
from utils.collate import collate_fn_base, collate_fn_tagger, collate_fn_joint, collate_fn_tagger_V2, collate_fn_bertbase_tti, collate_fn_tagger_V2TTI, collate_fn_jointV2, collate_fn_demo
from utils.convertor import switch_convertor
from utils.mask import convert_tagger2generator as tagger2generator
from utils.export import export_generator
from utils.data_utils import data_filter, reconstruct_switch, reconstruct_tagger, joint_report, map_unk2word, reconstruct_tagger_V2, fillin_tokens, report_pipeline_output, extract_generate_tokens, fillin_tokens4gts
from utils.data_utils import output_type_report
from utils.convertor import PointConverter, TaggerConverter
from utils.data_utils import obtain_uuid, convert_spmap_sw, convert_spmap2tokens, convert_spmap_tg
from utils.pipeline import base_context, split_sentence
from utils.defines import TEMPLATE_FILE_NAME, CACHE_DIR, SPLIT_VOCAB, INNER_VOCAB, BINARY_COLOR, UNDER_COLOR, TEXT_COLOR

AttnMask = SelfAttentionMask()

__all__ = ["get_device", "attention_mask", "padding", "norm_logits", "softmax_logits", "clip_maxgenerate", "save_model", "set_seed", "PointConverter",
           "TaggerConverter", "TextWash", "AttnMask", "logits_mask", "SwitchMetric", "SwitchSearch", "TaggerMetric", "tagger_loss_weights", "switch_convertor",
           "GeneratorMetric", "TAGGER_MAP", "TAGGER_MAP_INV", "INSERT_TAG", "MODIFY_TAG", "KEEP_TAG", "MOIFY_ONLY_TAG", "DELETE_TAG", "MODIFY_DELETE_TAG", "MASK_SYMBOL", "TYPE_MAP", "TYPE_MAP_INV",
           "collate_fn_base", "collate_fn_tagger", "collate_fn_joint", "collate_fn_tagger_V2", "collate_fn_bertbase_tti", "collate_fn_tagger_V2TTI", "collate_fn_jointV2", "collate_fn_demo",
           "tagger2generator", "export_generator", "fillin_tokens", "report_pipeline_output", "extract_generate_tokens", "fillin_tokens4gts",
           "data_filter", "reconstruct_switch", "reconstruct_tagger", "joint_report", "map_unk2word", "reconstruct_tagger_V2",
           "TaggerMetricV2",
           "output_type_report", "obtain_uuid", "convert_spmap_sw", "convert_spmap2tokens", "convert_spmap_tg",
           "base_context", "split_sentence", "TEMPLATE_FILE_NAME", "CACHE_DIR", "SPLIT_VOCAB", "INNER_VOCAB", "BINARY_COLOR", "UNDER_COLOR", "TEXT_COLOR"]