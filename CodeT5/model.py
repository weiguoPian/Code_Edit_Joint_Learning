import torch
import torch.nn as nn
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeT5_Model(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """

    def __init__(self, args, backbone, config):
        super(CodeT5_Model, self).__init__()
        self.backbone = backbone
        self.config = config

        self.args = args
        if self.args.localization:
            self.relu = nn.ReLU()
            self.pointer_projection = nn.Linear(config.hidden_size, config.hidden_size)
            self.pointer_out = nn.Linear(config.hidden_size, 20)
    
    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None, decoder_input_ids=None, decoder_attention_mask=None):
        if input_ids != None and encoder_outputs is None:
            output = self.backbone(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   decoder_input_ids=decoder_input_ids, 
                                   decoder_attention_mask=decoder_attention_mask)
            encoder_hidden = output.encoder_last_hidden_state[:,0,:]
            if self.args.localization:
                location_logits = self.pointer_out(self.relu(self.pointer_projection(encoder_hidden)))
                return output, location_logits
            else:
                return output

        elif input_ids is None and encoder_outputs != None:
            output = self.backbone(encoder_outputs=encoder_outputs,
                                   decoder_input_ids=decoder_input_ids,
                                   attention_mask=attention_mask)
            return output


    def get_encoder_out(self, input_ids=None, attention_mask=None):
        encoder = self.backbone.get_encoder()
        encoder_out = encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.args.localization:
            # print('encoder_out.last_hidden_state {}'.format(encoder_out.last_hidden_state.shape))
            location_logits = self.pointer_out(self.relu(self.pointer_projection(encoder_out.last_hidden_state[:,0,:])))
            return encoder_out, location_logits
        else:
            return encoder_out
    
    def backbone_config(self):
        return self.backbone.config