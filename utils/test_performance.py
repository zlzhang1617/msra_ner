def compare2tags(sent, gold, pre):
    length = len(sent)
    loc, per, org = [], [], []
    loc_, per_, org_ = [], [], []
    for i in range(length):
        if gold[i] == 'B-LOC':
            j = i + 1
            while j < length and gold[j] == 'I-LOC':
                j = j + 1
            loc.append(''.join(sent[i:j]))
        elif gold[i] == 'B-ORG':
            j = i + 1
            while j < length and gold[j] == 'I-ORG':
                j = j + 1
            org.append(''.join(sent[i:j]))
        elif gold[i] == 'B-PER':
            j = i + 1
            while j < length and gold[j] == 'I-PER':
                j = j + 1
            per.append(''.join(sent[i:j]))
    for i in range(length):
        if pre[i] == 'B-LOC':
            j = i + 1
            while j < length and pre[j] == 'I-LOC':
                j = j + 1
            loc_.append(''.join(sent[i:j]))
        elif pre[i] == 'B-ORG':
            j = i + 1
            while j < length and pre[j] == 'I-ORG':
                j = j + 1
            org_.append(''.join(sent[i:j]))
        elif pre[i] == 'B-PER':
            j = i + 1
            while j < length and pre[j] == 'I-PER':
                j = j + 1
            per_.append(''.join(sent[i:j]))
    a = len(loc) + len(per) + len(org)  # 文本中应有实体数
    b = len(loc_) + len(per_) + len(org_)  # 预测的实体数
    c = 0  # 预测正确的实体数
    if a == 0:
        if b == 0:
            return 1, 1
        else:
            return 0, 0
    if b == 0:
        if a == 0:
            return 1, 1
        else:
            return 0, 0
    for i in loc:
        if i in loc_: c = c + 1
    for i in org:
        if i in org_: c = c + 1
    for i in per:
        if i in per_: c = c + 1
    return c / b, c / a
