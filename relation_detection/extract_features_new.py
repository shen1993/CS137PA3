import re

import nltk
from nltk import Tree

"""
Notes: w_before_w1 and w_after_w2 seem to drop f1 by about 2 points....
"""
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

    def extract_sents(self, infilename, outfilename):
        with open(infilename, 'r') as input_f, open(outfilename, 'w') as raw_sents_f:
            lines = input_f.readlines()
            for line in lines:
                entries = line.split('\t')
                raw_sents = self.get_raw_sents(entries[1])
                print(raw_sents)


    def get_features(self, entries):
        # Write a list of features of the form "feat_name=value"
        raw_sents = self.get_raw_sents(entries[0])
        trees = self.get_trees(entries[0])
        ptrees = self.get_ptrees(entries[0])
        pos_list = self.get_pos(entries[0])
        w1_start = int(entries[2])
        w1_end = int(entries[3])
        w2_start = int(entries[8])
        w2_end = int(entries[9])
        betweenlist = raw_sents[int(entries[1])][w1_end:w2_start]
        wlist1 = raw_sents[int(entries[1])][w1_start:w1_end]
        wlist2 = raw_sents[int(entries[7])][w2_start:w2_end]
        pos1 = pos_list[int(entries[1])][w1_start:w1_end]
        pos2 = pos_list[int(entries[1])][w2_start:w2_end]

        name_ent1 = "NE1=" + entries[4].strip()
        tokens1 = "toks1=" + entries[6].strip()
        name_ent2 = "NE2=" + entries[10].strip()
        both_names = "bothNames=" + entries[4].strip() + '-' + entries[10].strip()
        tokens2 = "toks2=" + entries[12].strip()
        w_before_w1 = "wbefore_w1=" + self.word_before_w1(w1_start, raw_sents, entries)
        w_after_w2 = "wafter_w2=" + self.word_after_w2(w2_end, raw_sents, entries)
        # between word features
        if betweenlist:
            between_first = "between1=" + betweenlist[0]
            between_last = "between_end=" + betweenlist[-1]
            between_mid = "between_mid=" + '_'.join(betweenlist[1:-1])
        else:
            between_first = "between1=#adjacent"
            between_last = "between_end=#adjacent"
            between_mid = "between_mid=#adjacent"
        w_in_both = "shareword=" + self.common_words(wlist1, wlist2)
        between_len = "between_len=" + self.between_len_bin(betweenlist)
        head_head = "headhead=" + self.head_head(wlist1, wlist2)

        # These m1 in mention 2 dont' work well ...
        m1_in_m2_ent = "m1inm2_ent=" + self.X_in_Y(wlist1, wlist2) + both_names
        m1_in_m2_head ="m1inm2_head=" + self.X_in_Y(wlist1, wlist2) + head_head
        m2_in_m1_ent = "m2inm1_ent=" + self.X_in_Y(wlist2, wlist1) + both_names
        m2_in_m1_head = "m2inm1_head=" +self.X_in_Y(wlist2, wlist1) + head_head

        # Tree features
        # print("sent: ", raw_sents[int(entries[1])])
        # print("tree: ", trees[int(entries[1])])
        chunks = []
        self.chunkify(trees[int(entries[1])],chunks)


        # do chuks between, chunk heads

        # print(chunks)
        #clean_chunks = self.clean_chunks(chunks, raw_sents[int(entries[1])])
        # print('my chunking method: ', clean_chunks)
        # print('raw word2 : ', raw_sents[int(entries[1])][w2_start])

        # NP Chunks from regex parse in nltk
        word_pos = list(zip( [x for x in raw_sents[int(entries[1])] if x != '``' and x != '-LRB-'], pos_list[int(entries[1])]))

        # print('wordpos: ', word_pos)
        np_chunks = self.np_chunkify(word_pos)
        # print('npchunks: ', np_chunks)
        # print('raw word1 : ', raw_sents[int(entries[1])][w1_end-1])
        # print('between_chunks', self.get_chunks_between(w1_end, w2_start, clean_chunks))
        #between_chunks = self.get_chunks_between(w1_end, w2_start, clean_chunks)
        np_between_chunks = self.get_chunks_between(w1_end, w2_start,np_chunks)

        # Chunking features
        no_between_phrase = self.has_between_chunks(np_between_chunks)
        one_between_head = self.one_between_chunk(np_between_chunks)
        first_chunk_between = self.first_between_chunk(np_between_chunks)
        last_chunk_between = self.last_between_chunk(np_between_chunks)
        rest_between = self.rest_between_chunks(np_between_chunks)
        num_chunks_between = "numchunks=" + self.num_between(np_between_chunks)
        noun_level = self.noun_level(pos1, pos2)

        # print('sent: ', raw_sents[int(entries[1])])
        # print('tok1 tok2: ', tokens1, tokens2)
        # print('tree', ptrees[int(entries[1])])

        #Tree path
        tree_path_list = self.tree_path(w1_start, w1_end, w2_start, w2_end, ptrees[int(entries[1])])
        treepath = "treepath=" + '_'.join(tree_path_list)
        treepath_len = "treepath_len=" + str(len(tree_path_list))
        # print('tree: ', trees[int(entries[1])])
        # print('between: ', raw_sents[int(entries[1])][int(entries[3]):int(entries[8])])
        # print('wafter: ', w_after_w2)
        # print('wbefore: ', w_before_w1)

        # POS was not helpful at all
        # pos1 = "pos1=" + pos_list[int(entries[1])][int(entries[2]):int(entries[3])][0].strip()
        # pos2 = "pos2=" + pos_list[int(entries[7])][int(entries[8]):int(entries[9])][0].strip()

        nps_between = self.between_np_feature(raw_sents, pos_list, w1_end, w2_start, entries)
        vps_between = self.between_vp_feature(raw_sents, pos_list, w1_end, w2_start, entries)

        # Ablated
        #  nps_between, vps_between,  between_len

        return [name_ent1, tokens1, name_ent2, tokens2, both_names, between_first, between_last, between_mid, w_in_both,
                head_head,  noun_level, w_before_w1, w_after_w2, no_between_phrase, one_between_head,
                first_chunk_between, last_chunk_between, rest_between, num_chunks_between, nps_between, vps_between,  between_len,
               m2_in_m1_ent, m2_in_m1_head, m1_in_m2_ent, m1_in_m2_head, treepath_len, treepath]

    def between_vp_feature(self, raw_sents, pos_list, w1_end, w2_start, entries):
        vps_between = '_'.join(
            self.vps_between(pos_list[int(entries[1])], raw_sents[int(entries[1])], w1_end, w2_start))
        if vps_between:
            return "vps_between=" + vps_between
        else:
            return "vps_between=" + 'none'

    def between_np_feature(self, raw_sents, pos_list, w1_end, w2_start, entries):
        nps_between = '_'.join(
            self.nps_between(pos_list[int(entries[1])], raw_sents[int(entries[1])], w1_end, w2_start))
        if nps_between:
            return "nps_between=" + nps_between
        else:
            return "nps_between=" + 'none'

    def tree_path(self, w1_start, w1_end, w2_start, w2_end, tree):
        def get_lca_length(location1, location2):
            i = 0
            while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
                i += 1
            return i

        def get_labels_from_lca(ptree, lca_len, location):
            labels = []
            for i in range(lca_len, len(location)):
                labels.append(ptree[location[:i]].label())
            return labels

        def findPath(ptree, w1_start, w1_end, w2_start, w2_end):
            if w1_start != w2_end:
                location1 = ptree.treeposition_spanning_leaves(w1_start, w1_end)
            else:
                location1 = ptree.leaf_treeposition(w1_start)
            if w2_start != w2_end:
                location2 = ptree.treeposition_spanning_leaves(w2_start, w2_end)
            else:
                location2 = ptree.leaf_treeposition(w2_start)
            # find length of least common ancestor (lca)
            lca_len = get_lca_length(location1, location2)
            # find path from the node1 to lca
            labels1 = get_labels_from_lca(ptree, lca_len, location1)
            # ignore the first element, because it will be counted in the second part of the path
            result = labels1[1:]
            # inverse, because we want to go from the node to least common ancestor
            result = result[::-1]
            # add path from lca to node2
            result = result + get_labels_from_lca(ptree, lca_len, location2)
            return result[1:-1]  #weird error on the label of the two mentions pos, probably not necessary skip

        def normalize(path):
            prev = None
            ret = []
            for item in path:
                if prev != item:
                    ret.append(item)
                prev = item
            return ret

        return normalize(findPath(tree, w1_start, w1_end, w2_start, w2_end))


    def get_chunks_between(self, m1_end, m2_start, chunks):
        """Find a list of chunks in between 2 mention indeces"""
        len_chunks = []
        # print(chunks)
        for chunk in chunks:
            prev = len_chunks[-1] if len_chunks else 0
            len_chunks.append(len(chunk) + prev)
        chunk1_index = self.get_chunk_index(len_chunks, m1_end)
        chunk2_index = self.get_chunk_index(len_chunks, m2_start)
        return chunks[chunk1_index+1:chunk2_index+1]

    def get_chunk_index(self, len_chunks, index):
        """get the chunk index from word index"""
        i = 0
        # print('len_chunks', len_chunks)
        # print('index: ', index)
        # print(len_chunks)
        # print(index)
        while index > len_chunks[i] and i < len(len_chunks):
            # print('pre-i: ', i)
            i +=1
        # print('i: ', i)
        return i

    def chunkify(self, tree, chunks):
        chunk = []
        for child in tree:
            if isinstance(child, Tree):
                if child.label().startswith('S') or child.label() == 'NP' or child.label() == 'VP' or \
                                child.label() == 'PP' or child.label() == 'ADVP' or child.label() == 'ADJP':
                    chunks.append(chunk) # dump chunk, since interrupted
                    chunk = []           # make new chunk
                    self.chunkify(child, chunks)
                else:
                    #chunk.append(child.leaves())
                    #chunks.append(child.leaves())
                    chunk.extend(child.leaves())
            else:
                chunks.append([child])
        chunks.append(chunk)

    def clean_chunks(self, chunks, sent):
        """Fixes ordering in the chunks to match original sentence, strips empty lists"""
        chunks.sort(key= lambda x: len(x), reverse=True)
        fixed = []
        i = 0
        while i < len(sent):
            chunk_to_add = self.find_chunk(chunks, sent, i)
            if chunk_to_add:
                fixed.append(chunk_to_add)
                i += len(chunk_to_add)
            else:
                i += 1
        return fixed

    def find_chunk(self, chunks, sent, i):
        ret = None
        for chunk in chunks:
            if chunk:
                if self.chunk_match(chunk, sent[i: i+ len(chunk)]):
                    ret = chunk
                    chunks.remove(chunk)
                    return chunk
        return ret

    def chunk_match(self, chunk, subsent):
        if chunk is not None:
            for i in range(len(chunk)):
                if chunk[i] == subsent[i]:
                    pass
                else:
                    return False
            return True
        else:
            return False


    def X_in_Y(self, x, y):
        yset = set(y)
        for w in x:
            if w not in yset:
                return 'false'
        return 'true'

    def head_head(self, w1list, w2list):
        """The head isn't always the rightmost word, but most mentions are short enough for this to be tolerable"""
        w1_head = w1list[-1]
        w2_head = w2list[-1]
        return w1_head + '-' + w2_head

    def between_len_bin(self, betweenlist):
        length = len(betweenlist)
        if length == 0:
            return "0"
        elif length == 1:
            return "1"
        elif length == 2:
            return "2"
        elif length >2 and length <=6:
            return "few"
        elif length > 6 and length <=15:
            return "a_bunch"
        else:
            return "many"

    def word_before_w1(self, w1_start, raw_sents, entries):
        before_index = w1_start - 1
        if before_index >= 0:
            return raw_sents[int(entries[1])][before_index]
        else:
            return "none"

    def word_after_w2(self, w2_end, raw_sents, entries):
        after_index= w2_end
        if after_index < len(raw_sents[int(entries[1])]):
            return raw_sents[int(entries[1])][after_index]
        else:
            return "none"

    def common_words(self, wlist1, wlist2):
        set1 = set(wlist1)
        set2 = set(wlist2)
        intersection = set1.intersection(set2)
        return "true" if intersection else "false"

    def get_raw_sents(self, file_name):
        with open('data/parsed-files/' + file_name + '.head.rel.tokenized.raw.parse', 'r') as f:
            raw_sents = []
            for line in f:
                if line != '\n':
                    line = re.sub(r'\([^ ]+', '', line)
                    line = re.sub(r'\)', '', line)
                    line = re.sub(r'[ \t]+', ' ', line)
                    temp_list = line.replace('\n', '').split(' ')
                    sent_list = [x for x in temp_list if x != ''] #Harry's condition if x != '-LRB-' and x != ''
                    raw_sents.append(sent_list)
        return raw_sents

    def get_trees(self, file_name):
        with open('data/parsed-files/' + file_name + '.head.rel.tokenized.raw.parse', 'r') as f:
            trees = []
            for line in f:
                if line != '\n':
                    trees.append(Tree.fromstring(line))
        return trees

    def get_ptrees(self, file_name):
        with open('data/parsed-files/' + file_name + '.head.rel.tokenized.raw.parse', 'r') as f:
            trees = []
            for line in f:
                if line != '\n':
                    trees.append(nltk.ParentedTree.fromstring(line))
        return trees

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

    def has_between_chunks(self, between_chunks):
        if between_chunks:
            return "no_between=false"
        else:
            return "no_between=true"

    def one_between_chunk(self, between_chunks):
        if len(between_chunks) == 1:
            return "one_between=" + between_chunks[0][-1] #trivially the last word in chunk is head
        else:
            return "one_between=None"

    def first_between_chunk(self, between_chunks):
        if len(between_chunks) >= 2:
            return "first_between_chunk=" + between_chunks[0][-1] #head is last word
        else:
            return "first_between_chunk=none"

    def last_between_chunk(self, between_chunks):
        if len(between_chunks) >= 2:
            return "last_between_chunk=" + between_chunks[-1][-1]  # head is last word
        else:
            return "last_between_chunk=none"

    def rest_between_chunks(self, between_chunks):
        # assume the last word in chunk is the head
        if len(between_chunks) >= 3:
            return "rest_between_chunks=" + '_'.join([chunk[-1] for chunk in between_chunks[1:-1]])
        else:
            return "rest_between_chunks=none"

    def num_between(self, between_chunks):
        if len(between_chunks) == 0:
            return '0'
        elif len(between_chunks) == 1:
            return '1'
        elif len(between_chunks) == 2:
            return '2'
        elif len(between_chunks) > 2 and len(between_chunks) <= 6:
            return 'few'
        elif len(between_chunks) > 6 and len(between_chunks) <= 12:
            return 'many'
        else:
            return 'a_ton'

    def noun_level(self, pos1, pos2):
        return 'noun_level=' + pos1[-1] + '-' +  pos2[-1]

    def np_chunkify(self, word_pos):
        grammar = r"""
        NP: {<DT|POS\$>?<JJ>*<NN|NNP|NNS|NNPS>+}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}
      {<PRP>}
        """
        cp = nltk.RegexpParser(grammar)
        parsed = cp.parse(word_pos)
        chunks = []
        chunk = []
        for p in parsed:
            if isinstance(p, Tree):
                if chunk:
                    chunks.append(chunk)
                    chunk = []
                chunks.append([q[0] for q in p.leaves()])
            else:
                chunk.append(p[0])
        if chunk:
            chunks.append(chunk)
        return chunks

    def nps_between(self, pos, sent, w1, w2):
        posbetween = pos[w1:w2]
        sentbetween = sent[w1:w2]
        nounset = {'NN', 'NNP', 'NNPS', 'NNS', 'PRP'}
        return [sentbetween[i] for i in range(len(sentbetween)) if posbetween[i] in nounset]

    def vps_between(self, pos, sent, w1, w2):
        posbetween = pos[w1:w2]
        sentbetween = sent[w1:w2]
        return [sentbetween[i] for i in range(len(sentbetween)) if posbetween[i].startswith('V')]


if __name__ == '__main__':
    extractor = Extractor()
    extractor.extract_sents('data/rel-trainset.gold', 'trainset-raw-sents.txt')
    # extractor.extract('data/rel-trainset.gold', 'trainset-features-GOLD.txt', True)
    # extractor.extract('data/rel-devset.gold', 'devset-features-GOLD.txt', True)
    # extractor.extract('data/rel-devset.raw', 'devset-features-ONLY.txt', False)

    #Test set data
    #TODO run on test-set ?
    # extractor.extract('data/rel-testset.gold', 'testset-features-GOLD.txt', True)