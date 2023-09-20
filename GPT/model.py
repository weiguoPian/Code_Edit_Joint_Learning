# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class GPT_Model(nn.Module):
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
        super(GPT_Model, self).__init__()
        self.backbone = backbone
        self.config = config
        
        self.args = args
        if self.args.localization:
            self.relu = nn.ReLU()
            self.pointer_projection = nn.Linear(config.hidden_size, config.hidden_size)
            self.pointer_out = nn.Linear(config.hidden_size, 20)
    
    def forward(self, inputs, attention_mask=None, loss_mask=None, loc_attn_masks=None, past_key_values=None):
        if past_key_values != None:
            out = self.backbone(inputs, past_key_values=past_key_values)
            return out
        
        if attention_mask != None:
            outputs = self.backbone(inputs, attention_mask=attention_mask)

            if self.args.localization and loc_attn_masks != None:
                all_hiddens = outputs.hidden_states

                # (B, seq_len, hidden)
                last_layer_hidden = all_hiddens[-1]
                end_hidden = []
                pos_ids = torch.sum(loc_attn_masks, dim=-1) - 1
                for i in range(len(pos_ids)):
                    end_hidden.append(last_layer_hidden[i, pos_ids[i], :])
                # (B, hidden)
                end_hidden = torch.stack(end_hidden)
                location_logits = self.pointer_out(self.relu(self.pointer_projection(end_hidden)))
            
            logits = outputs[0]
            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])

            if self.args.localization:
                return loss, location_logits
            else:
                return loss
        else:
            outputs = self.backbone(inputs)
            out = outputs.past_key_values

            if loc_attn_masks is None:
                return out

            all_hiddens = outputs.hidden_states

            last_layer_hidden = all_hiddens[-1]
            end_hidden = []
            pos_ids = torch.sum(loc_attn_masks, dim=-1) - 1
            for i in range(len(pos_ids)):
                end_hidden.append(last_layer_hidden[i, pos_ids[i], :])
            end_hidden = torch.stack(end_hidden)
            location_logits = self.pointer_out(self.relu(self.pointer_projection(end_hidden)))

            if self.args.localization:
                return out, location_logits
            else:
                return out
