from seqeval.metrics import f1_score, precision_score, recall_score


class EDTester():
    def __init__(self, voc_i2s):
        self.voc_i2s = voc_i2s

    def calculate_report(self, y, y_, transform=True):
        '''
        calculating F1, P, R

        :param y: golden label, list
        :param y_: model output, list
        :return:
        '''
        if transform:
            for i in range(len(y)):
                for j in range(len(y[i])):
                    y[i][j] = self.voc_i2s[y[i][j]]
            for i in range(len(y_)):
                for j in range(len(y_[i])):
                    y_[i][j] = self.voc_i2s[y_[i][j]]
        return precision_score(y, y_), recall_score(y, y_), f1_score(y, y_)

    def calculate_report_id(self, y, y_, transform=True):
        '''
        calculating F1, P, R

        :param y: golden label, list
        :param y_: model output, list
        :return:
        '''
        if transform:
            for i in range(len(y)):
                for j in range(len(y[i])):
                    if y[i][j] != 'O':
                        y[i][j] = 'Trigger'
            for i in range(len(y_)):
                for j in range(len(y_[i])):
                    if y_[i][j] != 'O':
                        y_[i][j] = 'Trigger'
        return precision_score(y, y_), recall_score(y, y_), f1_score(y, y_)

    @staticmethod
    def merge_segments(y):
        segs = {}
        tt = ""
        st, ed = -1, -1
        for i, x in enumerate(y):
            if x.startswith("B-"):
                if tt == "":
                    tt = x[2:]
                    st = i
                else:
                    ed = i
                    segs[st] = (ed, tt)
                    tt = x[2:]
                    st = i
            elif x.startswith("I-"):
                if tt == "":
                    y[i] = "B" + y[i][1:]
                    tt = x[2:]
                    st = i
                else:
                    if tt != x[2:]:
                        ed = i
                        segs[st] = (ed, tt)
                        y[i] = "B" + y[i][1:]
                        tt = x[2:]
                        st = i
            else:
                ed = i
                if tt != "":
                    segs[st] = (ed, tt)
                tt = ""

        if tt != "":
            segs[st] = (len(y), tt)
        return segs

    def calculate_sets(self, y, y_):
        ct, p1, p2 = 0, 0, 0
        for sent, sent_ in zip(y, y_):
            for key, value in sent.items():
                p1 += len(value)
                if key not in sent_:
                    continue
                # matched sentences
                arguments = value
                arguments_ = sent_[key]
                ct += len(set(arguments) & set(arguments_))

            for key, value in sent_.items():
                p2 += len(value)

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1

    def calculate_sets_id(self, y, y_):
        ct, p1, p2 = 0, 0, 0
        for sent, sent_ in zip(y, y_):
            for key, value in sent.items():
                p1 += len(value)
                if key not in sent_:
                    continue
                # matched sentences
                arguments = value
                arguments_ = sent_[key]

                arguments_id = []
                arguments_id_ = []
                for arg in arguments:
                    arguments_id.append((arg[0], arg[1]))
                for arg_ in arguments_:
                    arguments_id_.append((arg_[0], arg_[1]))

                ct += len(set(arguments_id) & set(arguments_id_))

            for key, value in sent_.items():
                p2 += len(value)

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1