from collections import defaultdict

import torch
import torch.nn.functional as F
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.dataset_readers.dataset_utils.span_utils import \
    bio_tags_to_spans
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import contruct_overlap_graph


class BaseModel(nn.Module):

    def __init__(self, labels, model_name='allenai/scibert_scivocab_uncased', max_span_width=8):

        super().__init__()

        self.map_lab = dict((j, i) for i, j in enumerate(labels, start=1))

        self.max_span_width = max_span_width

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(self, tokens, tags=None):

        if tags is None:
            tags = ['O' for _ in range(len(tokens))]

        def span_to_label(tags):
            """Convert tags to spans

            Args:
                tags (list[str]): BIO tags

            Returns:
                defaultdict: Mapping span idx to labels
            """

            dict_tag = defaultdict(int)

            spans_with_labels = bio_tags_to_spans(tags)

            for label, span in spans_with_labels:
                dict_tag[span] = self.map_lab[label]

            return dict_tag

        tokenized = [self.tokenizer.encode(
            t, add_special_tokens=False) for t in tokens]  # tokenization

        tokenized_2 = []
        tags_2 = []
        subword_lenghts = []
        i=0
        for tok, tg in zip(tokenized, tags):
            subword_lenghts.append(i)
            i += len(tok)
            if i < 512:
                tokenized_2.append(tok)
                tags_2.append(tg)

        # compute span from BIO
        dict_map = span_to_label(tags_2)  # dict containing 'span'->'label'
        
        # all possible spans
        span_ids = enumerate_spans(
            tokenized_2, max_span_width=self.max_span_width)

        # span lengths
        span_lengths = []
        for idxs in span_ids:
            sid, eid = idxs
            slen = eid - sid
            span_lengths.append(slen)

        # sword boundary => span boundary
        mapping = dict(zip(range(len(subword_lenghts)), subword_lenghts))
        subword_boundaries = list(
            map(lambda x: (mapping[x[0]], mapping[x[1]]), span_ids))

        # span labels
        span_labels = torch.LongTensor(
            [dict_map[i] for i in span_ids]
        )

        original_spans = torch.LongTensor(span_ids)  # [num_spans, 2]

        input_ids, span_ids, span_lengths = map(torch.LongTensor, [
                                                [i for k in tokenized for i in k], subword_boundaries, span_lengths])

        # graph construction
        inter_mask = contruct_overlap_graph(span_ids)

        return {'input_ids': input_ids, 'span_ids': span_ids, 'span_lengths': span_lengths, 'span_labels': span_labels, 'inter_mask': inter_mask, 'original_spans': original_spans}

    def collate_fn(self, batch_list):

        # preprocess batch
        batch = [self.preprocess(tokens, tags) for tokens, tags in batch_list]

        # Char batch
        input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)  # [B, W, C]

        # span mask
        span_ids = pad_sequence(
            [b['span_ids'] for b in batch], batch_first=True, padding_value=-1)

        span_mask = span_ids.sum(-1) != -2

        span_ids = span_ids.masked_fill(span_ids == -1, 0)

        original_spans = pad_sequence(
            [b['original_spans'] for b in batch], batch_first=True, padding_value=0
        )

        # attention_mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        # span label
        span_labels = pad_sequence(
            [b['span_labels'] for b in batch], batch_first=True, padding_value=-1
        )

        span_lengths = pad_sequence(
            [b['span_lengths'] for b in batch], batch_first=True, padding_value=0
        )

        # Interaction mask
        graph_overlap = [b['inter_mask'] for b in batch]

        max_size = max([i.size(0) for i in graph_overlap])

        padded_graph = torch.zeros(len(graph_overlap), max_size, max_size)

        for id, m in enumerate(graph_overlap):
            size = m.size(0)
            padded_graph[id, :size, :size] = m

        return {'input_ids': input_ids, 'span_ids': span_ids, 'attention_mask': attention_mask, 'span_labels': span_labels, 'span_mask': span_mask, 'span_lengths': span_lengths, 'graph': padded_graph, 'original_spans': original_spans}

    def create_dataloader(self, data, **kwargs):
        return DataLoader(data, collate_fn=self.collate_fn, **kwargs)
