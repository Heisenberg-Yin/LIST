#revised from https://github.com/microsoft/AR2/tree/main/AR2
from src.rank.model.hfencoder import HFBertEncoder
from torch import nn

    
class BiBertEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, args):
        super(BiBertEncoder, self).__init__()
        self.question_model = HFBertEncoder.init_encoder(args, args.bert_dropout)
        # Check if weights should be shared between question and context models
        if hasattr(args, 'share_weight') and args.share_weight:
            self.ctx_model = self.question_model
        else:
            self.ctx_model = HFBertEncoder.init_encoder(args, args.bert_dropout)
        
    def query_emb(self, input_ids, token_type_ids, attention_mask):
        # Get query embeddings using the question model
        _, pooled_output, _ = self.question_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return pooled_output

    def body_emb(self, input_ids, token_type_ids, attention_mask):
        # Get context embeddings using the context model
        _, pooled_output, _ = self.ctx_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return pooled_output