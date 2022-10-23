from config.switch_config import parse_args as switch_parse
from config.tagger_config import parse_args as tagger_parse
from config.generator_config import parse_args as generator_parse
from config.evaluate_indep_config import parse_args as evalindep_parse
from config.joint_config import parse_args as joint_parse
from config.evaluate_joint_config import parse_args as evaljoint_parse

__all__ = ["switch_parse", "tagger_parse", "generator_parse", "evalindep_parse", "joint_parse", "evaljoint_parse"]