import torch
from sklearn.metrics import f1_score


def extract_spans(predictions, span_mask, span_idx):
    """Extract spans + labels

    Args:
        predictions (torch.Tensor): predictions [B, num_spans]
        span_mask (torch.Tensor): mask per batch [B, num_spans]
        span_idx (torch.Tensor): spans [B, num_spans]

    Returns:
        [type]: [description]
    """

    lengths = span_mask.sum(-1)  # [B,]

    all_pred = []

    for pred, l, span in zip(predictions, lengths, span_idx):

        pred = pred[:l]

        idx = torch.where(pred > 0)[0]

        # number if non-O spans
        if idx.nelement() == 0:
            all_pred.append([])
            continue

        idx = idx.tolist()

        spans = []

        for i in idx:
            # start, end
            spi = span[i].tolist()

            # append label
            spi.append(pred[i].item())

            # to tuple
            spans.append(tuple(spi))

        all_pred.append(spans)

    return all_pred


def validate(true, pred, mask):
    """validate true and pred
    Args:
        true : [Batch_size, num_spans]
        pred : [Batch_size, num_spans]
        mask : [Batch_size, num_spans]

    Returns:
        masked true and pred
    """

    true = true.view(-1)
    pred = pred.view(-1)
    mask = mask.view(-1)

    valid = (true != 0) + (pred != 0)
    valid = valid * mask

    true = true[valid].tolist()
    pred = pred[valid].tolist()

    return true, pred


def compute_metrics(true, pred, mask, labels=None):
    """Compute f-score

    Args:
        true ([type]): [description]
        pred ([type]): [description]
        mask ([type]): [description]
        labels ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    true, pred = validate(true, pred, mask)

    micro = f1_score(true, pred, labels=labels, average='micro')
    macro = f1_score(true, pred, labels=labels, average='macro')

    return {'f1_micro': micro, 'f1_macro': macro, 'true': true, 'pred': pred}


def contruct_overlap_graph(span_idx):
    num_spans = len(span_idx)
    interaction_mask = torch.zeros(num_spans, num_spans)
    for i in range(num_spans):
        for j in range(num_spans):
            s_i = span_idx[i].tolist()
            s_j = span_idx[j].tolist()
            if has_overlapping(s_i, s_j):
                interaction_mask[i, j] = -1
            if i == j:
                interaction_mask[i, j] = 1
    return interaction_mask


def has_overlapping(idx1, idx2):
    overlapping = True
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]):
        overlapping = False
    return overlapping


def num_ovelap_span(result):
    num_ov = 0
    for res in result:
        if len(res) < 2:
            continue
        for i in range(len(res) - 1):
            for j in range(i+1, len(res)):
                num_ov += has_overlapping(res[i], res[j])
    return num_ov
