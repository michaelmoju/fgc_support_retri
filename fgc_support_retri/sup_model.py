from enum import Enum
from transformers import BertModel
import torch
import torch.nn as nn


class BertSupSentClassification(nn.Module):
	class ForwardMode(Enum):
		TRAIN = 0
		EVAL = 1

	def __init__(self, bert_encoder: BertModel):
		super(BertSupSentClassification, self).__init__()
		self.bert_encoder = bert_encoder
		self.dropout = nn.Dropout(p=bert_encoder.config.hidden_dropout_prob)
		self.classifier = nn.Linear(bert_encoder.config.hidden_siz, 1)
		self.criterion = nn.BCEWithLogitsLoss()

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, mode=ForwardMode.TRAIN, labels=None):
		sequence_output, pooled_output = self.bert_encoder(input_ids, token_type_ids, attention_mask)
		hidden = self.dropout(pooled_output)
		logits = self.classifier(hidden)

		if mode == BertSupSentClassification.ForwardMode.TRAIN:
			loss = self.criterion(logits, labels.unsqueeze(-1).float())
			return loss

		elif mode == BertSupSentClassification.ForwardMode.EVAL:
			return logits
        
        
class BertForMultiHopQuestionAnswering(nn.Module):
    class ForwardMode(Enum):
        TRAIN = 0
        EVAL = 1
    
    def __init__(self, bert_encoder: BertModel):
        super(BertForMultiHopQuestionAnswering, self).__init__()
#         self.bert = BertModelPlus(config)
        self.bert_encoder = bert_encoder
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
#         self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        sep_positions=None,
        hop_start_weights=None,
        hop_end_weights=None,
        ans_start_weights=None,
        ans_end_weights=None,
        B_starts=None,
        allow_limit=(0, 0),
    ):
        """ Extract spans by System 1.
        
        Args:
            input_ids (LongTensor): Token ids of word-pieces. (batch_size * max_length)
            token_type_ids (LongTensor): The A/B Segmentation in BERTs. (batch_size * max_length)
            attention_mask (LongTensor): Indicating whether the position is a token or padding. (batch_size * max_length)
            sep_positions (LongTensor): Positions of [SEP] tokens, mainly used in finding the num_sen of supporing facts. (batch_size * max_seps)
            hop_start_weights (Tensor): The ground truth of the probability of hop start positions. The weight of sample has been added on the ground truth. 
                (You can verify it by examining the gradient of binary cross entropy.)
            hop_end_weights ([Tensor]): The ground truth of the probability of hop end positions.
            ans_start_weights ([Tensor]): The ground truth of the probability of ans start positions.
            ans_end_weights ([Tensor]): The ground truth of the probability of ans end positions.
            B_starts (LongTensor): Start positions of sentence B.
            allow_limit (tuple, optional): An Offset for negative threshold. Defaults to (0, 0).
        
        Returns:
            [type]: [description]
        """
        batch_size = input_ids.size()[0]
        device = input_ids.get_device() if input_ids.is_cuda else torch.device("cpu")
        sequence_output, hidden_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        semantics = hidden_output[:, 0]
        # Some shapes: sequence_output [batch_size, max_length, hidden_size], pooled_output [batch_size, hidden_size]
        if sep_positions is None:
            return semantics  # Only semantics, used in bundle forward
        else:
            max_sep = sep_positions.size()[-1]
        if max_sep == 0:
            empty = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return (
                empty,
                empty,
                semantics,
                empty,
            )  # Only semantics, used in eval, the same ``empty'' variable is a mistake in general cases but simple

        # Predict spans
        logits = self.qa_outputs(sequence_output)
        hop_start_logits, hop_end_logits, ans_start_logits, ans_end_logits = logits.split(1, dim=-1)
        hop_start_logits = hop_start_logits.squeeze(-1)
        hop_end_logits = hop_end_logits.squeeze(-1)
        ans_start_logits = ans_start_logits.squeeze(-1)
        ans_end_logits = ans_end_logits.squeeze(-1)  # Shape: [batch_size, max_length]

        if hop_start_weights is not None:  # Train mode
            lgsf = torch.nn.LogSoftmax(
                dim=1
            )  # If there is no targeted span in the sentence, start_weights = end_weights = 0(vec)
            hop_start_loss = -torch.sum(
                hop_start_weights * lgsf(hop_start_logits), dim=-1
            )
            hop_end_loss = -torch.sum(hop_end_weights * lgsf(hop_end_logits), dim=-1)
            ans_start_loss = -torch.sum(
                ans_start_weights * lgsf(ans_start_logits), dim=-1
            )
            ans_end_loss = -torch.sum(ans_end_weights * lgsf(ans_end_logits), dim=-1)
            hop_loss = torch.mean((hop_start_loss + hop_end_loss)) / 2
            ans_loss = torch.mean((ans_start_loss + ans_end_loss)) / 2
        else:
            # In eval mode, find the exact top K spans.
            K_hop, K_ans = 3, 1
            hop_preds = torch.zeros(
                batch_size, K_hop, 3, dtype=torch.long, device=device
            )  # (start, end, sen_num)
            ans_preds = torch.zeros(
                batch_size, K_ans, 3, dtype=torch.long, device=device
            )
            ans_start_gap = torch.zeros(batch_size, device=device)
            for u, (start_logits, end_logits, preds, K, allow) in enumerate(
                (
                    (
                        hop_start_logits,
                        hop_end_logits,
                        hop_preds,
                        K_hop,
                        allow_limit[0],
                    ),
                    (
                        ans_start_logits,
                        ans_end_logits,
                        ans_preds,
                        K_ans,
                        allow_limit[1],
                    ),
                )
            ):
                for i in range(batch_size):
                    if sep_positions[i, 0] > 0:
                        values, indices = start_logits[i, B_starts[i] :].topk(K)
                        for k, index in enumerate(indices):
                            if values[k] <= start_logits[i, 0] - allow:  # not golden
                                if u == 1: # For ans spans
                                    ans_start_gap[i] = start_logits[i, 0] - values[k]
                                break
                            start = index + B_starts[i]
                            # find ending
                            for j, ending in enumerate(sep_positions[i]):
                                if ending > start or ending <= 0:
                                    break
                            if ending <= start:
                                break
                            ending = min(ending, start + 10)
                            end = torch.argmax(end_logits[i, start:ending]) + start
                            preds[i, k, 0] = start
                            preds[i, k, 1] = end
                            preds[i, k, 2] = j
        return (
            (hop_loss, ans_loss, semantics)
            if hop_start_weights is not None
            else (hop_preds, ans_preds, semantics, ans_start_gap)
        )


if __name__ == '__main__':
	pass


