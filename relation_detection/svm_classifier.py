import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def loading_data(file_path):
    return_me = []
    with open(file_path) as file:
        line = file.readline()
        while line:
            item = tuple(line.replace('\n', '').split('\t'))
            if item != ('',):
                return_me.append(item)
            line = file.readline()
    return return_me


def get_features(sent):
    toks1 = sent[2].replace('toks1=', '')
    toks2 = sent[4].replace('toks2=', '')
    ne1 = sent[1].replace('NE1=', '')
    ne2 = sent[3].replace('NE2=', '')

    # Combined name entities
    bothNE = sent[5].replace('bothNames=', '')

    # 1st, last, rest of word between mentions
    between1 = sent[6].replace('between1=', '')
    between_end = sent[7].replace('between_end=', '')
    between_mid = sent[8].replace('between_mid=', '')

    shareword = sent[9].replace('shareword=', '')

    # Head-head
    headhead = sent[10].replace('headhead=', '')

    # Noun-Noun level
    noun_level = sent[11].replace('noun_level=', '')

    wbefore_w1 = sent[12].replace('wbefore_w1=', '')
    wafter_w2 = sent[13].replace('wafter_w2=', '')

    # Chunking
    no_between = sent[14].replace('no_between=', '')
    one_between = sent[15].replace('one_between=', '')
    first_between_chunk = sent[16].replace('first_between_chunk=', '')
    last_between_chunk = sent[17].replace('last_between_chunk=', '')
    rest_between_chunks = sent[18].replace('rest_between_chunks=', '')
    numchunks = sent[19].replace('numchunks=', '')

    # Between
    nps_between = sent[20].replace('nps_between=', '')
    vps_between = sent[21].replace('vps_between=', '')
    between_len = sent[22].replace('between_len=', '')

    # Overlaps
    m2inm1_ent = sent[23].replace('m2inm1_ent=', '')
    m2inm1_head = sent[24].replace('m2inm1_head=', '')
    m1inm2_ent = sent[25].replace('m1inm2_ent=', '')
    m1inm2_head = sent[26].replace('m1inm2_head=', '')
    if m2inm1_ent[0:5] == 'false':
        m2inm1 = 'false'
    elif m2inm1_ent[0:4] == 'true':
        m2inm1 = 'true'
    else:
        print("ERROR!")
    if m1inm2_ent[0:5] == 'false':
        m1inm2 = 'false'
    elif m1inm2_ent[0:4] == 'true':
        m1inm2 = 'true'
    else:
        print("ERROR!")

    # Trees
    treepath_len = sent[27].replace('treepath_len=', '')
    treepath = sent[28].replace('treepath=', '')

    features = {
        'WM1': toks1.lower(),
        'WM2': toks2.lower(),
        'ET1': ne1,
        'ET2': ne2,
        'ET12': bothNE,
        'between1': between1.lower(),
        'between_end': between_end.lower(),
        'between_mid': between_mid.lower(),
        'shareword': shareword,

        # 'headhead': headhead.lower(),

        'noun_level': noun_level,

        'wbefore_w1': wbefore_w1.lower(),
        'wafter_w2': wafter_w2.lower(),

        'no_between': no_between,
        'one_between': one_between.lower(),
        'first_between_chunk': first_between_chunk.lower(),
        'last_between_chunk': last_between_chunk.lower(),
        'rest_between_chunks': rest_between_chunks.lower(),
        'numchunks': numchunks,

        'nps_between': nps_between.lower(),
        'vps_between': vps_between.lower(),
        'between_len': between_len,

        # 'M1>M2':m2inm1,
        # 'M2>M1':m1inm2,
        # 'ET12+M1>M2': m2inm1_ent.lower(),
        # 'HM12+M1>M2': m2inm1_head.lower(),
        # 'ET12+M1<M2': m1inm2_ent.lower(),
        # 'HM12+M1<M2': m1inm2_head.lower(),

        # 'treepath_len': treepath_len,
        # 'treepath': treepath
    }
    return features


def get_labels(sent):
    return sent[0]


def get_result_score(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    test_total = 0
    gold_total = 0
    correct = 0

    for i in range(len(y_test)):
        if y_test[i] != 'no_rel':
            gold_total += 1
        if y_pred[i] != 'no_rel':
            test_total += 1
        if y_test[i] != 'no_rel' and y_test[i] == y_pred[i]:
            correct += 1

    precision = float(correct) / test_total
    recall = float(correct) / gold_total
    f = precision * recall * 2 / (precision + recall)

    print('precision =', precision, 'recall =', recall, 'f1 =', f)
    return y_pred


train_list = loading_data('trainset-features-GOLD.txt')
dev_list = loading_data('devset-features-GOLD.txt')

print("Feature example", get_features(train_list[0]))
vec = DictVectorizer()

X_train = vec.fit_transform([get_features(s) for s in train_list])
y_train = [get_labels(s) for s in train_list]

X_dev = vec.transform([get_features(s) for s in dev_list])
y_dev = [get_labels(s) for s in dev_list]

start_time = time.time()
print("Start training...")
clf = OneVsRestClassifier(LinearSVC(multi_class='ovr')).fit(X_train, y_train)
print("Finished. Time: {0:.2g}s".format(time.time() - start_time))

get_result_score(clf, X_dev, y_dev)

final_test = True
if final_test:
    test_list = loading_data('testset-features-GOLD.txt')
    X_test = vec.transform([get_features(s) for s in test_list])
    y_test = [get_labels(s) for s in test_list]
    y_pred = get_result_score(clf, X_test, y_test)
    with open("rel-testset.svm", 'w') as output_f, open("data/rel-testset.raw", 'r') as input_f:
        lines = input_f.readlines()
        for i, line in enumerate(lines):
            output_f.write(y_pred[i] + '\t' + line)
