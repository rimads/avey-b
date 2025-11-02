from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .configuration_sl_model import SLModelConfig
from .modeling_sl_model import (
    SLModel,
    SLModelForQuestionAnswering,
    SLModelForSequenceClassification,
    SLModelForTokenClassification,
)


AutoConfig.register("sl_model", SLModelConfig)
AutoModel.register(SLModelConfig, SLModel)
AutoModelForQuestionAnswering.register(SLModelConfig, SLModelForQuestionAnswering)
AutoModelForSequenceClassification.register(SLModelConfig, SLModelForSequenceClassification)
AutoModelForTokenClassification.register(SLModelConfig, SLModelForTokenClassification)

SLModelConfig.register_for_auto_class()
SLModel.register_for_auto_class("AutoModel")
SLModelForQuestionAnswering.register_for_auto_class("AutoModelForQuestionAnswering")
SLModelForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
SLModelForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
