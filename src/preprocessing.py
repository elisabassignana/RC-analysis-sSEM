import json
from torch.utils.data import Dataset, DataLoader
from itertools import permutations
from src.utils.bucketing import Buckets


class DatasetMapper(Dataset):

    def __init__(self, sentences, entities_1, entities_2, relations):
        self.sentences = sentences
        self.entities_1 = entities_1
        self.entities_2 = entities_2
        self.relations = relations

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.entities_1[idx], self.entities_2[idx], self.relations[idx]

class DatasetMapperDoubleMarkers(Dataset):

    def __init__(self, sentences, entities_1, entities_2, relations, entities_1end, entities_2end):
        self.sentences = sentences
        self.entities_1 = entities_1
        self.entities_2 = entities_2
        self.relations = relations
        self.entities_1end = entities_1end
        self.entities_2end = entities_2end

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.entities_1[idx], self.entities_2[idx], self.relations[idx], self.entities_1end[idx], self.entities_2end[idx]

def prepare_data_train(data_path, labels2id, batch_size, repeat_entities_markers, end_entities_markers):

    sentences_tot, entities_1_tot, entities_2_tot, relations_tot = read_json_file_crossre(data_path, labels2id, repeat_entities_markers, end_entities_markers)

    data_loader = DataLoader(DatasetMapper(sentences_tot, entities_1_tot, entities_2_tot, relations_tot), batch_size=batch_size)
    return data_loader, relations_tot

def prepare_data_test(data_path, batch_size, buckets, repeat_entities_markers, end_entities_markers):

    # these are list of list
    sentences_tot, entities_1_tot, entities_2_tot, relations_tot = read_json_file_crossre_eval(data_path, buckets, repeat_entities_markers, end_entities_markers)

    data_loader = []
    for i in range(len(sentences_tot)):
        # one DataLoader for each bucket
        data_loader.append(DataLoader(DatasetMapper(sentences_tot[i], entities_1_tot[i], entities_2_tot[i], relations_tot[i]), batch_size=batch_size))

    return data_loader, relations_tot

def prepare_data_train_DoubleMarkers(data_path, labels2id, batch_size):

    sentences_tot, entities_1_tot, entities_2_tot, relations_tot, entities_1end_tot, entities_2end_tot = read_json_file_crossre_DoubleMarkers(data_path, labels2id)
    data_loader = DataLoader(DatasetMapperDoubleMarkers(sentences_tot, entities_1_tot, entities_2_tot, relations_tot, entities_1end_tot, entities_2end_tot), batch_size=batch_size)

    return data_loader, relations_tot

def prepare_data_test_DoubleMarkers(data_path, batch_size, buckets):

    sentences_tot, entities_1_tot, entities_2_tot, relations_tot, entities_1end_tot, entities_2end_tot = read_json_file_crossre_eval_DoubleMarkers(data_path, buckets)
    data_loader = []
    for i in range(len(sentences_tot)):
        # one DataLoader for each bucket
        data_loader.append(DataLoader(DatasetMapperDoubleMarkers(sentences_tot[i], entities_1_tot[i], entities_2_tot[i], relations_tot[i], entities_1end_tot[i], entities_2end_tot[i]), batch_size=batch_size))

    return data_loader, relations_tot

# return sentences, idx within the sentence of entity-markers-start, relation labels
def read_json_file_crossre_eval(json_file, buckets, repeat_entities_markers=False, end_entities_markers=False):

    with open(json_file) as data_file:
        for json_elem in data_file:
            abstract = json.loads(json_elem)

            # set json instance in bucket class
            if buckets.attribute != 'none':
                buckets.set_json_instance(abstract)

            # consider only the sentences with at least 2 entities
            if len(abstract["ner"]) > 1:

                # create all the possible entity pairs
                entity_pairs = permutations(abstract["ner"], 2)

                for entity_pair in entity_pairs:

                    # set the entity tokens to inject in the instance
                    ent1_start = f'<E1>'
                    ent1_end = f'</E1>'
                    ent2_start = f'<E2>'
                    ent2_end = f'</E2>'

                    # build the instance sentence for the model
                    sentence_marked = ''

                    for idx_token in range(len(abstract["sentence"])):

                        # nested entities begin
                        if idx_token == entity_pair[0][0] and idx_token == entity_pair[1][0]:
                            # entity 1 is the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][1] > entity_pair[1][1]:
                                sentence_marked += f'{ent1_start} {ent2_start} {abstract["sentence"][idx_token]} '
                                # entity 2 (the shortest one) is one token long
                                if idx_token == entity_pair[1][1]:
                                    sentence_marked += f'{ent2_end} '
                            # entity 2 is the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{ent2_start} {ent1_start} {abstract["sentence"][idx_token]} '
                                # entity 1 (the shortest one) is one token long
                                if idx_token == entity_pair[0][1]:
                                    sentence_marked += f'{ent1_end} '

                        # match begin entity 1
                        elif idx_token == entity_pair[0][0]:
                            sentence_marked += f'{ent1_start} {abstract["sentence"][idx_token]} '
                            # entity 1 is one token long
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '
                            # entity 1 is a nested entity encapsulated inside entity 2
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                        # match begin entity 2
                        elif idx_token == entity_pair[1][0]:
                            sentence_marked += f'{ent2_start} {abstract["sentence"][idx_token]} '
                            # entity 2 is one token long
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                            # entity 2 is a nested entity encapsulated inside entity 1
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '

                        # nested entities end
                        elif idx_token == entity_pair[0][1] and idx_token == entity_pair[1][1]:
                            # entity 1 in the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][0] < entity_pair[1][0]:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} {ent1_end} '
                            # entity 2 in the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} {ent2_end} '

                        # match end entity 1
                        elif idx_token == entity_pair[0][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} '
                        # match end entity 2
                        elif idx_token == entity_pair[1][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} '

                        # regular token
                        else:
                            sentence_marked += f'{abstract["sentence"][idx_token]} '

                    # retrieve relation label
                    dataset_relations = [(e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) for (e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) in abstract["relations"] if e1_s == entity_pair[0][0] and e1_e == entity_pair[0][1] and e2_s == entity_pair[1][0] and e2_e == entity_pair[1][1]]

                    # compute current attribute value/s
                    buckets.compute_current_attribute(entity_pair, [elem[4] for elem in dataset_relations])

                    # prepare data
                    if len(dataset_relations) > 0:
                        # prepare sentence
                        sentence_marked = sentence_marked.strip()
                        # strategy repeat entities at the end
                        if repeat_entities_markers or end_entities_markers:
                            # retrieve entity tokens
                            entity_tokens_1 = ''
                            for i in range(entity_pair[0][0], entity_pair[0][1] + 1):
                                entity_tokens_1 += f'{abstract["sentence"][i]} '
                            entity_tokens_2 = ''
                            for i in range(entity_pair[1][0], entity_pair[1][1] + 1):
                                entity_tokens_2 += f'{abstract["sentence"][i]} '

                            # concatenate entities at the end of the sentence
                            sentence_marked += f'[SEP] <E1> {entity_tokens_1.strip()} </E1> <E2> {entity_tokens_2.strip()} </E2>'

                        # prepare entity markers position
                        if end_entities_markers:
                            # retrieve both, store last instance
                            indices_e1, indices_e2 = [], []
                            for idx, value in enumerate(sentence_marked.split(' ')):
                                if value == '<E1>':
                                    indices_e1.append(idx)
                                if value == '<E2>':
                                    indices_e2.append(idx)
                            position_e1 = indices_e1[1]
                            position_e2 = indices_e2[1]
                        else:
                            position_e1 = sentence_marked.split(' ').index(f'{ent1_start}')
                            position_e2 = sentence_marked.split(' ').index(f'{ent2_start}')

                        # update the buckets
                        buckets.update_buckets(sentence_marked,
                                               position_e1,
                                               position_e2,
                                               [elem[4] for elem in dataset_relations])

    return buckets.get_data_buckets()

# return sentences, idx within the sentence of entity-markers-start, relation labels
def read_json_file_crossre(json_file, labels2id, repeat_entities_markers, end_entities_markers):

    sentences, entities_1, entities_2, relations = [], [], [], []

    with open(json_file) as data_file:
        for json_elem in data_file:
            abstract = json.loads(json_elem)

            # consider only the sentences with at least 2 entities
            if len(abstract["ner"]) > 1:

                # create all the possible entity pairs
                entity_pairs = permutations(abstract["ner"], 2)

                for entity_pair in entity_pairs:

                    # set the entity tokens to inject in the instance
                    ent1_start = f'<E1>'
                    ent1_end = f'</E1>'
                    ent2_start = f'<E2>'
                    ent2_end = f'</E2>'

                    # build the instance sentence for the model
                    sentence_marked = ''

                    for idx_token in range(len(abstract["sentence"])):

                        # nested entities begin
                        if idx_token == entity_pair[0][0] and idx_token == entity_pair[1][0]:
                            # entity 1 is the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][1] > entity_pair[1][1]:
                                sentence_marked += f'{ent1_start} {ent2_start} {abstract["sentence"][idx_token]} '
                                # entity 2 (the shortest one) is one token long
                                if idx_token == entity_pair[1][1]:
                                    sentence_marked += f'{ent2_end} '
                            # entity 2 is the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{ent2_start} {ent1_start} {abstract["sentence"][idx_token]} '
                                # entity 1 (the shortest one) is one token long
                                if idx_token == entity_pair[0][1]:
                                    sentence_marked += f'{ent1_end} '

                        # match begin entity 1
                        elif idx_token == entity_pair[0][0]:
                            sentence_marked += f'{ent1_start} {abstract["sentence"][idx_token]} '
                            # entity 1 is one token long
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '
                            # entity 1 is a nested entity encapsulated inside entity 2
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                        # match begin entity 2
                        elif idx_token == entity_pair[1][0]:
                            sentence_marked += f'{ent2_start} {abstract["sentence"][idx_token]} '
                            # entity 2 is one token long
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                            # entity 2 is a nested entity encapsulated inside entity 1
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '

                        # nested entities end
                        elif idx_token == entity_pair[0][1] and idx_token == entity_pair[1][1]:
                            # entity 1 in the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][0] < entity_pair[1][0]:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} {ent1_end} '
                            # entity 2 in the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} {ent2_end} '

                        # match end entity 1
                        elif idx_token == entity_pair[0][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} '
                        # match end entity 2
                        elif idx_token == entity_pair[1][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} '

                        # regular token
                        else:
                            sentence_marked += f'{abstract["sentence"][idx_token]} '

                    # retrieve relation label
                    dataset_relations = [(e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) for (e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) in abstract["relations"] if e1_s == entity_pair[0][0] and e1_e == entity_pair[0][1] and e2_s == entity_pair[1][0] and e2_e == entity_pair[1][1]]

                    # prepare data
                    if len(dataset_relations) > 0:
                        # gold label
                        relations.append(labels2id[dataset_relations[0][4]])

                        # sentence
                        if repeat_entities_markers or end_entities_markers:
                            # retrieve entity tokens
                            entity_tokens_1 = ''
                            for i in range(entity_pair[0][0], entity_pair[0][1] + 1):
                                entity_tokens_1 += f'{abstract["sentence"][i]} '
                            entity_tokens_2 = ''
                            for i in range(entity_pair[1][0], entity_pair[1][1] + 1):
                                entity_tokens_2 += f'{abstract["sentence"][i]} '

                            # concatenate entities at the end of the sentence
                            sentence_marked = sentence_marked.strip()
                            sentence_marked += f'[SEP] <E1> {entity_tokens_1.strip()} </E1> <E2> {entity_tokens_2.strip()} </E2>'
                        sentences.append(sentence_marked.strip())

                        # position of entity markers
                        if end_entities_markers:
                            # retrieve both, store last instance
                            indices_e1, indices_e2 = [], []
                            for idx, value in enumerate(sentence_marked.split(' ')):
                                if value == '<E1>':
                                    indices_e1.append(idx)
                                if value == '<E2>':
                                    indices_e2.append(idx)
                            entities_1.append(indices_e1[1])
                            entities_2.append(indices_e2[1])
                        else:
                            entities_1.append(sentence_marked.split(' ').index(f'{ent1_start}'))
                            entities_2.append(sentence_marked.split(' ').index(f'{ent2_start}'))

    return sentences, entities_1, entities_2, relations

def read_json_file_crossre_eval_DoubleMarkers(json_file, buckets):

    with open(json_file) as data_file:
        for json_elem in data_file:
            abstract = json.loads(json_elem)

            # set json instance in bucket class
            if buckets.attribute != 'none':
                buckets.set_json_instance(abstract)

            # consider only the sentences with at least 2 entities
            if len(abstract["ner"]) > 1:

                # create all the possible entity pairs
                entity_pairs = permutations(abstract["ner"], 2)

                for entity_pair in entity_pairs:

                    # set the entity tokens to inject in the instance
                    ent1_start = f'<E1>'
                    ent1_end = f'</E1>'
                    ent2_start = f'<E2>'
                    ent2_end = f'</E2>'

                    # build the instance sentence for the model
                    sentence_marked = ''

                    for idx_token in range(len(abstract["sentence"])):

                        # nested entities begin
                        if idx_token == entity_pair[0][0] and idx_token == entity_pair[1][0]:
                            # entity 1 is the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][1] > entity_pair[1][1]:
                                sentence_marked += f'{ent1_start} {ent2_start} {abstract["sentence"][idx_token]} '
                                # entity 2 (the shortest one) is one token long
                                if idx_token == entity_pair[1][1]:
                                    sentence_marked += f'{ent2_end} '
                            # entity 2 is the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{ent2_start} {ent1_start} {abstract["sentence"][idx_token]} '
                                # entity 1 (the shortest one) is one token long
                                if idx_token == entity_pair[0][1]:
                                    sentence_marked += f'{ent1_end} '

                        # match begin entity 1
                        elif idx_token == entity_pair[0][0]:
                            sentence_marked += f'{ent1_start} {abstract["sentence"][idx_token]} '
                            # entity 1 is one token long
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '
                            # entity 1 is a nested entity encapsulated inside entity 2
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                        # match begin entity 2
                        elif idx_token == entity_pair[1][0]:
                            sentence_marked += f'{ent2_start} {abstract["sentence"][idx_token]} '
                            # entity 2 is one token long
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                            # entity 2 is a nested entity encapsulated inside entity 1
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '

                        # nested entities end
                        elif idx_token == entity_pair[0][1] and idx_token == entity_pair[1][1]:
                            # entity 1 in the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][0] < entity_pair[1][0]:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} {ent1_end} '
                            # entity 2 in the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} {ent2_end} '

                        # match end entity 1
                        elif idx_token == entity_pair[0][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} '
                        # match end entity 2
                        elif idx_token == entity_pair[1][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} '

                        # regular token
                        else:
                            sentence_marked += f'{abstract["sentence"][idx_token]} '

                    # retrieve relation label
                    dataset_relations = [(e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) for (e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) in abstract["relations"] if e1_s == entity_pair[0][0] and e1_e == entity_pair[0][1] and e2_s == entity_pair[1][0] and e2_e == entity_pair[1][1]]

                    # compute current attribute value/s
                    buckets.compute_current_attribute(entity_pair, [elem[4] for elem in dataset_relations])

                    # prepare data
                    if len(dataset_relations) > 0:
                        # prepare sentence
                        sentence_marked = sentence_marked.strip()
                        entity_tokens_1 = ''
                        for i in range(entity_pair[0][0], entity_pair[0][1] + 1):
                            entity_tokens_1 += f'{abstract["sentence"][i]} '
                        entity_tokens_2 = ''
                        for i in range(entity_pair[1][0], entity_pair[1][1] + 1):
                            entity_tokens_2 += f'{abstract["sentence"][i]} '
                        # sentence
                        sentence_marked += f'[SEP] <E1> {entity_tokens_1.strip()} </E1> <E2> {entity_tokens_2.strip()} </E2>'

                        # prepare entity markers position
                        indices_e1, indices_e2 = [], []
                        for idx, value in enumerate(sentence_marked.split(' ')):
                            if value == '<E1>':
                                indices_e1.append(idx)
                            if value == '<E2>':
                                indices_e2.append(idx)

                        # update the buckets
                        buckets.update_buckets_double_markers(sentence_marked,
                                                              indices_e1[0],
                                                              indices_e2[0],
                                                              [elem[4] for elem in dataset_relations],
                                                              indices_e1[1],
                                                              indices_e2[1]
                                                              )

    return buckets.get_data_buckets()

def read_json_file_crossre_DoubleMarkers(json_file, labels2id):

    sentences, entities_1, entities_2, entities_1end, entities_2end, relations = [], [], [], [], [], []

    with open(json_file) as data_file:
        for json_elem in data_file:
            abstract = json.loads(json_elem)

            # consider only the sentences with at least 2 entities
            if len(abstract["ner"]) > 1:

                # create all the possible entity pairs
                entity_pairs = permutations(abstract["ner"], 2)

                for entity_pair in entity_pairs:

                    # set the entity tokens to inject in the instance
                    ent1_start = f'<E1>'
                    ent1_end = f'</E1>'
                    ent2_start = f'<E2>'
                    ent2_end = f'</E2>'

                    # build the instance sentence for the model
                    sentence_marked = ''

                    for idx_token in range(len(abstract["sentence"])):

                        # nested entities begin
                        if idx_token == entity_pair[0][0] and idx_token == entity_pair[1][0]:
                            # entity 1 is the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][1] > entity_pair[1][1]:
                                sentence_marked += f'{ent1_start} {ent2_start} {abstract["sentence"][idx_token]} '
                                # entity 2 (the shortest one) is one token long
                                if idx_token == entity_pair[1][1]:
                                    sentence_marked += f'{ent2_end} '
                            # entity 2 is the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{ent2_start} {ent1_start} {abstract["sentence"][idx_token]} '
                                # entity 1 (the shortest one) is one token long
                                if idx_token == entity_pair[0][1]:
                                    sentence_marked += f'{ent1_end} '

                        # match begin entity 1
                        elif idx_token == entity_pair[0][0]:
                            sentence_marked += f'{ent1_start} {abstract["sentence"][idx_token]} '
                            # entity 1 is one token long
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '
                            # entity 1 is a nested entity encapsulated inside entity 2
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                        # match begin entity 2
                        elif idx_token == entity_pair[1][0]:
                            sentence_marked += f'{ent2_start} {abstract["sentence"][idx_token]} '
                            # entity 2 is one token long
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                            # entity 2 is a nested entity encapsulated inside entity 1
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '

                        # nested entities end
                        elif idx_token == entity_pair[0][1] and idx_token == entity_pair[1][1]:
                            # entity 1 in the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][0] < entity_pair[1][0]:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} {ent1_end} '
                            # entity 2 in the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} {ent2_end} '

                        # match end entity 1
                        elif idx_token == entity_pair[0][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} '
                        # match end entity 2
                        elif idx_token == entity_pair[1][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} '

                        # regular token
                        else:
                            sentence_marked += f'{abstract["sentence"][idx_token]} '

                    # retrieve relation label
                    dataset_relations = [(e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) for (e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) in abstract["relations"] if e1_s == entity_pair[0][0] and e1_e == entity_pair[0][1] and e2_s == entity_pair[1][0] and e2_e == entity_pair[1][1]]

                    # prepare data
                    if len(dataset_relations) > 0:
                        # gold label
                        relations.append(labels2id[dataset_relations[0][4]])

                        # retrieve entity tokens
                        entity_tokens_1 = ''
                        for i in range(entity_pair[0][0], entity_pair[0][1] + 1):
                            entity_tokens_1 += f'{abstract["sentence"][i]} '
                        entity_tokens_2 = ''
                        for i in range(entity_pair[1][0], entity_pair[1][1] + 1):
                            entity_tokens_2 += f'{abstract["sentence"][i]} '
                        # sentence
                        sentence_marked += f'[SEP] <E1> {entity_tokens_1.strip()} </E1> <E2> {entity_tokens_2.strip()} </E2>'
                        sentences.append(sentence_marked.strip())

                        # position of entity markers
                        indices_e1, indices_e2 = [], []
                        for idx, value in enumerate(sentence_marked.split(' ')):
                            if value == '<E1>':
                                indices_e1.append(idx)
                            if value == '<E2>':
                                indices_e2.append(idx)
                        entities_1.append(indices_e1[0])
                        entities_1end.append(indices_e1[1])
                        entities_2.append(indices_e2[0])
                        entities_2end.append(indices_e2[1])


    return sentences, entities_1, entities_2, relations, entities_1end, entities_2end

