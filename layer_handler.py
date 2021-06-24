import torch
import torch.nn as nn

class Bert_Layer_Handler():
    '''
    Allows you to get the outputs at a BERT layer of a trained BertGrader

    AND has a separate method to pass an embedding through
    and remaining layers of the model and further for BERTGrader

    Bert model layers 1-12
    layer = 13 is interpreted as pooler output
    layer = 14 is interpreted as logits output
    '''

    def __init__(self, trained_model, layer_num=1):
        trained_model.eval()
        self.model = trained_model
        self.layer_num = layer_num

    def get_layern_outputs(self, input_ids, attention_mask, device=torch.device('cpu')):
        '''
        Get output hidden states from nth layer
        '''
        self.model.to(device)

        if self.layer_num == 13:
            # Need pooler output
            output = self.model.bert(input_ids, attention_mask)
            return output.pooler_output
        if self.layer_num == 14:
            # Need logits output
            return self.model(input_ids, attention_mask)

        # Need to extend mask for encoder - from HuggingFace implementation
        self.input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.model.bert.get_extended_attention_mask(attention_mask, self.input_shape, device)

        hidden_states = self.model.bert.embeddings(input_ids=input_ids)
        for layer_module in self.model.bert.encoder.layer[:self.layer_num]:
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
        return hidden_states

    def pass_through_rest(self, hidden_states, attention_mask, device=torch.device('cpu')):
        '''
        Pass hidden states through remainder of BertGrader model
        after nth layer

        Not defined for layer_num=14
        '''
        if self.layer_num == 13:
            # hidden states should be the pooler output
            logits = self.model.classifier(hidden_states)
            return logits

        extended_attention_mask: torch.Tensor = self.model.bert.get_extended_attention_mask(attention_mask, self.input_shape, device)

        for layer_module in self.model.bert.encoder.layer[self.layer_num:]:
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]

        sentence_embedding = self.model.bert.pooler(hidden_states)
        logits = self.model.classifier(sentence_embedding)
        return logits
