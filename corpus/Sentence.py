from enet.consts import CUTOFF
import copy


def pretty_str(a):
    a = a.upper()
    if a == 'O':
        return a
    elif a[1] == '-':
        return a[:2] + "|".join(a[2:].split("-")).replace(":", "||")
    else:
        return "|".join(a.split("-")).replace(":", "||")


class Sentence:
    def __init__(self, json_content, graph_field_name="stanford-colcc"):
        self.wordList = json_content["words"][:CUTOFF]
        self.posLabelList = json_content["pos-tags"][:CUTOFF]
        self.lemmaList = json_content["lemma"][:CUTOFF]
        self.length = len(self.wordList)

        self.entityLabelList = self.generateEntityLabelList(json_content["golden-entity-mentions"])
        self.triggerLabelList = self.generateTriggerLabelList(json_content["golden-event-mentions"])

        # external knowledge graph
        self.externalNodeList = json_content["external-nodes"]
        self.externalNodes_length = len(self.externalNodeList)
        self.allNodeList = self.wordList + self.externalNodeList
        self.allNodes_length = len(self.allNodeList)
        self.sentenceMap = json_content["sentence-map"]
        self.adjpos, self.adjv = self.generateGraphMatrix(json_content["graph-triples"], json_content["external-nodes"])

        self.entities = self.generateGoldenEntities(json_content["golden-entity-mentions"])
        self.events = self.generateGoldenEvents(json_content["golden-event-mentions"])

        self.containsEvents = len(json_content["golden-event-mentions"])
        self.tokenList = self.makeTokenList()


    def generateEntityLabelList(self, entitiesJsonList):
        '''
        Keep the overlapping entity labels

        [['B-PER'], ['I-PER'], ['I-PER'], ['I-PER', 'B-PER'], ['I-PER', 'I-PER'], ['I-PER', 'I-PER'], ['O'],
         ['O'], ['O'], ['O'], ['B-FAC'], ['I-FAC'], ['B-GPE', 'I-FAC'], ['I-FAC'], ['I-FAC', 'B-FAC'],
         ['B-PER', 'I-FAC', 'B-PER'], ['I-PER', 'I-FAC', 'I-PER', 'B-TIM'], ['I-PER', 'I-FAC', 'I-PER'],
         ['I-FAC', 'I-PER'], ['I-FAC', 'I-PER'], ['I-FAC', 'I-PER'], ['I-FAC'], ['I-FAC'], ['I-FAC'], ['I-FAC'],
         ['I-FAC'], ['I-FAC'], ['I-FAC'], ['I-FAC', 'B-TIM'], ['I-FAC', 'I-TIM'], ['O'], ['B-PER'], ['O'], ['B-TIM'], ['O']]

        :param entitiesJsonList:
        :return:
        '''
        entityLabel = [["O"] for _ in range(self.length)]

        def assignEntityLabel(index, label):
            if index >= CUTOFF:
                return
            if len(entityLabel[index]) == 1 and entityLabel[index][0] == "O":
                entityLabel[index][0] = pretty_str(label)
            else:
                entityLabel[index].append(pretty_str(label))

        for entityJson in entitiesJsonList:
            start = entityJson["start"]
            end = entityJson["end"]
            etype = entityJson["entity-type"].split(":")[0]
            assignEntityLabel(start, "B-" + etype)
            for i in range(start + 1, end):
                assignEntityLabel(i, "I-" + etype)

        return entityLabel


    def generateGoldenEntities(self, entitiesJson):
        '''
        [(2, 3, "entity_type")]
        '''
        golden_list = []
        for entityJson in entitiesJson:
            start = entityJson["start"]
            if start >= CUTOFF:
                continue
            end = min(entityJson["end"], CUTOFF)
            etype = entityJson["entity-type"].split(":")[0]
            golden_list.append((start, end, etype))
        return golden_list


    def generateGoldenEvents(self, eventsJson):
        '''
        {
            (19, 20, 'LIFE||DIE'): [(14, 15, 'PLACE'), (15, 21, 'VICTIM'), (28, 30, 'TIME|BEFORE')]
        }

        {
            (2, 3, "event_type_str") --> [(1, 2, "role_type_str"), ...]
            ...
        }

        '''
        golden_dict = {}
        for eventJson in eventsJson:
            triggerJson = eventJson["trigger"]
            if triggerJson["start"] >= CUTOFF:
                continue
            key = (triggerJson["start"], min(triggerJson["end"], CUTOFF), pretty_str(eventJson["event_type"]))
            values = []
            for argumentJson in eventJson["arguments"]:
                if argumentJson["start"] >= CUTOFF:
                    continue
                value = (argumentJson["start"], min(argumentJson["end"], CUTOFF), pretty_str(argumentJson["role"]))
                values.append(value)
            golden_dict[key] = list(sorted(values))
        return golden_dict


    def generateTriggerLabelList(self, triggerJsonList):
        '''

        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LIFE||DIE',
         'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        :param triggerJsonList:
        :return:
        '''
        triggerLabel = ["O" for _ in range(self.length)]

        def assignTriggerLabel(index, label):
            if index >= CUTOFF:
                return
            triggerLabel[index] = pretty_str(label)

        for eventJson in triggerJsonList:
            triggerJson = eventJson["trigger"]
            start = triggerJson["start"]
            end = triggerJson["end"]
            etype = eventJson["event_type"]
            assignTriggerLabel(start, "B-" + etype)
            for i in range(start + 1, end):
                assignTriggerLabel(i, "I-" + etype)
        return triggerLabel


    def generateGraphMatrix(self, triplesList, externalNodesList):
        sparseAdjMatrixPos = [[], [], []]
        sparseAdjMatrixValues = []

        def addedge(type_, from_, to_, value_):
            sparseAdjMatrixPos[0].append(type_)
            sparseAdjMatrixPos[1].append(from_)
            sparseAdjMatrixPos[2].append(to_)
            sparseAdjMatrixValues.append(value_)

        def isInt(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        def generateExternalAdjMatrix(triplesList, externalNodesList):
            for triple in triplesList:
                ss = triple.split("///")
                edge_weight = float(ss[3])
                if (isInt(ss[0])):
                    fromIndex = int(ss[0])
                    if (isInt(ss[1])):
                        if ss[2] in externalNodesList:
                            toIndex = self.length + externalNodesList.index(ss[2])
                            addedge(int(ss[1]), fromIndex, toIndex, edge_weight)
                        else:
                            print(ss)
                            print("[Error in Sentence.py] external knowledge node (in lexical level) not in externalNodesList: ", ss[2])
                    else:
                        toIndex = int(ss[2])
                        if ss[1] == 'B':
                            addedge(3, fromIndex, toIndex, edge_weight)
                            addedge(4, toIndex, fromIndex, edge_weight)
                        elif ss[1] == 'I':
                            addedge(5, fromIndex, toIndex, edge_weight) 
                else:
                    if all(n in externalNodesList for n in [ss[0], ss[2]]):
                        fromIndex = self.length + externalNodesList.index(ss[0])
                        toIndex = self.length + externalNodesList.index(ss[2])
                        addedge(3, fromIndex, toIndex, edge_weight)
                        addedge(4, toIndex, fromIndex, edge_weight)
                    else:
                        print(ss)
                        print("[Error in Sentence.py] external knowledge node (in sentence level) not in externalNodesList: ", ss[0], ss[2])

        generateExternalAdjMatrix(triplesList, externalNodesList)
        return sparseAdjMatrixPos, sparseAdjMatrixValues


    def makeTokenList(self):
        return [Token(self.wordList[i], self.posLabelList[i], self.lemmaList[i], self.entityLabelList[i], self.triggerLabelList[i])
                for i in range(self.length)]

    def __len__(self):
        return self.length

    def __iter__(self):
        for x in self.tokenList:
            yield x

    def __getitem__(self, index):
        return self.tokenList[index]


class Token:
    def __init__(self, word, posLabel, lemmaLabel, entityLabel, triggerLabel):
        self.word = word
        self.posLabel = posLabel
        self.lemmaLabel = lemmaLabel
        self.entityLabel = entityLabel
        self.triggerLabel = triggerLabel
        self.predictedLabel = None

    def addPredictedLabel(self, label):
        self.predictedLabel = label
