#revised from https://github.com/microsoft/AR2/tree/main/AR2
import transformers
from transformers import (
    BertModel,
    BertConfig,
)
import logging
logger = logging.getLogger(__name__)

class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
        self.version = int(transformers.__version__.split('.')[0])
        
    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1, model_type=None):
        if model_type is None:
            model_type = args.model_type
        cfg = BertConfig.from_pretrained(model_type)
        # dropout = args.bert_dropout
        # logger.info("dropout: ", dropout)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(transformers.__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing
        return cls.from_pretrained(model_type, config=cfg)

    def forward(self, **kwargs):
        hidden_states = None
        result = super().forward(**kwargs)
        sequence_output = result.last_hidden_state + 0 * result.pooler_output.sum()
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states