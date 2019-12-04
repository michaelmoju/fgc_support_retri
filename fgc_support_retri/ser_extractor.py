import torch

class SER_extract:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_name = 'bert-base-chinese'
        bert_encoder = BertModel.from_pretrained(bert_model_name)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_indexer = BertIdx(bert_tokenizer)
        model = BertSupSentClassification(bert_encoder)
#         model_path = config.TRAINED_MODELS / '20191129-with_hotpot'/ 'model_epoch5_loss_0.226.m'
        model_path = config.TRAINED_MODELS / '20191128'/ 'model_epoch5_loss_0.213.m' 
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = model
        self.bert_indexer = bert_indexer
        self.device = device
        
    def predict(self, context_sents, question):
        batch = []
        for sent in context_sents:
            sample = self.bert_indexer({'QTEXT':question, 'sentence': sent})
            batch.append(sample)
        batch = bert_collate(batch)      
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            logits = self.model(input_ids=input_ids, 
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask, 
                                mode=BertSupSentClassification.ForwardMode.EVAL)
        return logits
    
if __name__ == '__main__':
    ser_extracter = SER_extract()