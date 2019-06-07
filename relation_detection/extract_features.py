import re


class Extractor:
    def __init__(self):
        pass

    def extract(self, infilename, outfilename, gold=True):
        with open(infilename, 'r') as input_f, open(outfilename, 'w') as feature_f:
            lines = input_f.readlines()
            for line in lines:
                entries = line.split('\t')
                outlist = []
                if gold:
                    outlist.append(entries[0])
                    outlist.extend(self.get_features(entries[1:]))
                else:
                    outlist.extend(self.get_features(entries))
                feature_f.write('\t'.join(outlist) + '\n')

    def get_features(self, entries):
        # Write a list of features of the form "feat_name=value"
        # TODO open pos and tree files to get more advanced features
        raw_sents = self.get_raw_sents(entries[0])
        pos_list = self.get_pos(entries[0])
        # print(raw_sents[int(entries[1])][int(entries[2]):int(entries[3])])
        # print(pos_list[int(entries[1])][int(entries[2]):int(entries[3])])
        name_ent1 = "NE1=" + entries[4]
        tokens1 = "toks1=" + entries[6]
        name_ent2 = "NE2=" + entries[10]
        tokens2 = "toks2=" + entries[12]
        return [name_ent1, tokens1, name_ent2, tokens2]

    def get_raw_sents(self, file_name):
        with open('data/parsed-files/' + file_name + '.head.rel.tokenized.raw.parse', 'r') as f:
            raw_sents = []
            for line in f:
                if line != '\n':
                    line = re.sub(r'\([^ ]+', '', line)
                    line = re.sub(r'\)', '', line)
                    line = re.sub(r'[ \t]+', ' ', line)
                    temp_list = line.replace('\n', '').split(' ')
                    sent_list = [x for x in temp_list if x != '-LRB-' and x != '']
                    raw_sents.append(sent_list)
        return raw_sents

    def get_pos(self, file_name):
        with open('data/postagged-files/' + file_name + '.head.rel.tokenized.raw.tag', 'r') as f:
            pos_list = []
            for line in f:
                if line != '\n':
                    line = ' ' + line
                    line = re.sub(r' [^_]+_', ' ', line)
                    temp_list = line.replace('\n', '').split(' ')
                    sent_list = [x for x in temp_list if x != '']
                    pos_list.append(sent_list)
        return pos_list


if __name__ == '__main__':
    extractor = Extractor()
    extractor.extract('data/rel-trainset.gold', 'trainset-features-GOLD.txt', True)
    # extractor.extract('data/rel-devset.raw', 'devset-features-ONLY.txt', False)
    # extractor.extract('data/rel-devset.gold', 'devset-features-GOLD.txt', True)
