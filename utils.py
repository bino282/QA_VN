import re
from collections import Counter
from keras.preprocessing import sequence

def clean_text(text):
    text = text.lower()
    text = re.sub('[^_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ0-9a-z.]',' ',text)
    text = text.split()
    text = [w for w in text if not w.isdigit()]
    return text

def read_data_from_file(path_name):
    questions = []
    answers = []
    labels = []
    with open(path_name,'r',encoding='utf-8') as lines:
        for line in lines:
            tmp = line.strip().split('\t')
            questions.append(clean_text(tmp[0]))
            answers.append(clean_text(tmp[1]))
            labels.append(int(tmp[2]))
    return questions,answers,labels

def creat_voc(data,min_count = 5):
    voc = set()
    all_words = []
    for sent in data:
        for w in sent:
            voc.add(w)
            all_words.append(w)
    counter = Counter(all_words)
    voc = list(voc)
    voc = [w for w in voc if counter[w] > min_count]
    voc.insert(0,'<PAD>')
    voc.insert(1,'<UNK>')
    voc2index = {}
    for i in range(len(voc)):
        voc2index[voc[i]] = i
    return voc,voc2index
def convert_and_pad(X,voc2index,max_len):
    X_idx = [convert_data_to_index(x,voc2index) for x in X]
    X_pad = sequence.pad_sequences(X_idx,maxlen=max_len,padding='post',truncating="post")
    return X_pad

def convert_data_to_index(string_data, vocab):
    index_data = []
    for i in range(len(string_data)):
        if string_data[i] in vocab:
            index_data.append(vocab[string_data[i]])
    return index_data

def map_score(s1,s2,y_pred,labels_dev):
    QA_pairs = {}
    for i in range(len(s1)):
        pred = y_pred[i]

        s1_str = " ".join(s1[i])
        s2_str = " ".join(s2[i])
        if s1 in QA_pairs:
            QA_pairs[s1_str].append((s2_str, labels_dev[i], pred[1]))
        else:
            QA_pairs[s1_str] = [(s2_str, labels_dev[i], pred[1])]

    MAP, MRR = 0, 0
    num_q = len(QA_pairs.keys())
    for s1 in QA_pairs.keys():
        p, AP = 0, 0
        MRR_check = False

        QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)

        for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
            if int(label) == 1:
                if not MRR_check:
                    MRR += 1 / (idx + 1)
                    MRR_check = True

                p += 1
                AP += p / (idx + 1)
        if(p==0):
            AP = 0
        else:
            AP /= p
        MAP += AP
    MAP /= num_q
    MRR /= num_q
    return MAP,MRR
