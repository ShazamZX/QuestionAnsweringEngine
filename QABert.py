import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
class QABert:
    def __init__(self):
       self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
       self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    def pred_answer(self, question, context):
        input_ids = self.tokenizer.encode(question,context)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        #first occurence of [SEP] token
        sep_idx = input_ids.index(self.tokenizer.sep_token_id)
        #number of tokens in segment A (question)
        num_seg_a = sep_idx+1
        #number of tokens in segment B (text)
        num_seg_b = len(input_ids) - num_seg_a
        #list of 0s and 1s for segment embeddings
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(input_ids)
        
        output = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
        
        #reconstructing the answer
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        answer = ""
        if answer_end >= answer_start:
            answer = tokens[answer_start]
            for i in range(answer_start+1, answer_end+1):
                if tokens[i][0:2] == "##":
                    answer += tokens[i][2:]
                else:
                    answer += " " + tokens[i]
                    
        if answer.startswith("[CLS]"):
            answer = "Unable to find the answer to your question."
        if answer.endswith("[SEP]"):
            answer = answer[:-5]
        
        return answer.capitalize()