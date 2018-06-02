# в этом скртипте всякие вспомогательные функции для обработки

import wget
import numpy as np
import msgpack
from nltk.tokenize import RegexpTokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import FastText
import fastText
from fastText import load_model
from tqdm import tqdm_notebook as tqdm

from constants import *


def get_json_data(download=False):
    PATH = 'train-v1.1.json'
    if (download):
        wget.download('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json', out=PATH)
    with open(PATH) as lines:
        data = json.loads(lines.read())
    return data



def tokenize(text, lower=True):
    tokenizer = RegexpTokenizer('\w+|\$[\d]+|\S+')
    return [token.lower() for token in tokenizer.tokenize(text) if token.isalnum()]


# def normalize(text, lower=True):
#     return ' '.join(tokenize(text, lower))


def normalize(data):
    return (" ".join(data))


def find_span(text, answer):
    text = normalize(text)
    answer = normalize(answer)

    index = text.find(answer)
    if index == -1:
        end = len(tokenize(text))
        return end, end

    left_part = text[:index]
    start = len(tokenize(left_part))
    end = start + len(tokenize(answer))

    return start, end


#back
def decode(sequence, back):
    return np.array([back[i] for i in sequence])

def process_data(data):
    model = load_model("wiki.simple.bin")
    model_s = list()
    dic = {}
    back = {}
    t = 1
    zero = np.zeros(300)
    model_s.append(zero)
    model_s.append(zero)
    print(model_s[0])
    dic['$'] = 0 #end
    dic['&'] = 1 #none
    back[0] = '$'
    back[1] = '&'
    
    max_par_size = 0
    max_que_size = 0

    contexts_input = list()
    contexts_len = list()
    questions_input = list()
    questions_len = list()
    ans_start = list()
    ans_end = list()

    for i in tqdm(range(len(data['data']))):
        # проход по всем статьям википедии
        article = data['data'][i]

        # проход по всем параграфам в данной статье
        for parag in article['paragraphs']:
            # сюда будем пихать численное описание вектора
            context_vector = list()

            context = parag['context']
            t_context = tokenize(context)

            for i in range(len(t_context)):
                if t_context[i] in dic:
                    x = dic[t_context[i]]
                else:
                    t += 1
                    dic[t_context[i]] = t
                    back[t] = t_context[i]
                    # тут надо еще обыграть что может не быть такого вектора
                    # !!!
                    model_s.append(model.get_word_vector(t_context[i]))
                    x = t
                context_vector.append(x)

            context_size = len(context_vector)

            # проход по всем вопросам к этому парарграфу    
            for qn in parag['qas']:
                question = qn['question']
                t_question = tokenize(question)

                quest_vector = list()

                for j in range(len(t_question)):
                    if t_question[j] in dic:
                        x = dic[t_question[j]]
                    else:
                        t += 1
                        dic[t_question[j]] = t
                        back[t] = t_question[j]
                        # тут надо еще обыграть что может не быть такого вектора
                        # !!!
                        model_s.append(model.get_word_vector(t_question[j]))
                        x = t
                    quest_vector.append(x)

                quest_size = len(quest_vector)

                answer = qn['answers'][0]['text']
                start, end = find_span(context, answer)

                #Добавим в массив:

                contexts_input.append(context_vector)
                contexts_len.append(context_size)
                questions_input.append(quest_vector)
                questions_len.append(quest_size)
                ans_start.append(start)
                ans_end.append(end)

                max_que_size = max(quest_size, max_que_size)

            max_par_size = max(context_size, max_par_size)
            #paragraphs_ft.append(vec)
            
    return model_s, dic, back, max_par_size, max_que_size, contexts_input, contexts_len, questions_input, questions_len, ans_start, ans_end,
        
        
def shuffled_index(sizes, bucket_size=10000):
    size = len(sizes)
    index = list(range(size))

    def key(i):
        return sizes[i]

    index.sort(key=key)

    for start in range(0, size, bucket_size):
        end = min(start + bucket_size, size)
        random.shuffle(index[start: end])
    return index


def simple_pad(data, size):
    for text in data:
        i = len(text)
        while (i < size + 1):
            text.append(1)
            i += 1
    return data


def embedder(data, model_s):
    output = np.zeros((len(data), len(data[0]), 300))
    for i in range(len(data)):
        for j in range(len(data[0])):
            
            output[i][i] = model_s[data[i][j]]
    return output

def batch_gen(pivot, data, meta, coords, train_flag=True):
    
    data_size = len(data)
    shuffled_indexes = np.arange(0, data_size, 1)
    np.random.shuffle(shuffled_indexes)
    
    num_batches = data_size // pivot
    for i in range(num_batches):
        cur_indexes = shuffled_indexes[i * pivot:(i + 1) * pivot]
        context_embedded = np.zeros((pivot, C_MAX_SIZE, EMBEDDING_SIZE))
        context_len = np.zeros((pivot), dtype=np.int32)
        features = np.zeros((pivot, C_MAX_SIZE, FEATURES_SIZE), dtype=np.float64)

        question_embedded = np.zeros((pivot, Q_MAX_SIZE, EMBEDDING_SIZE))
        masked_question = np.zeros((pivot), dtype=np.int32)

        starts = np.zeros((pivot))
        ends = np.zeros((pivot))

        for i in range(pivot):

            question = data[shuffled_indexes[i]][5] 
            masked_question[i] = len(question)

            for j in range(len(question)):
                question_embedded[i][j] = meta['embedding'][question[j]]
            for j in range(len(question), Q_MAX_SIZE):
                question_embedded[i][j] = meta['embedding'][0]

            if train_flag:
                starts[i] = data[shuffled_indexes[i]][8]
                ends[i] = data[shuffled_indexes[i]][9]
            else:
                starts[i] = coords[shuffled_indexes[i]][0]
                ends[i] = coords[shuffled_indexes[i]][1]

            paragraph = data[shuffled_indexes[i]][1]

            context_len[i] = len(paragraph)
            for j in range(len(paragraph)):
                context_embedded[i][j] = meta['embedding'][paragraph[j]]
                features[i][j][:len(data[shuffled_indexes[i]][2][j])] = data[shuffled_indexes[i]][2][j]
                features[i][j][len(data[shuffled_indexes[i]][2][j]) + data[shuffled_indexes[i]][4][j]] = 1
                features[i][j][len(data[shuffled_indexes[i]][2][j]) + ent_num + data[shuffled_indexes[i]][3][j]] = 1
            for j in range(len(paragraph), C_MAX_SIZE):
                context_embedded[i][j] = meta['embedding'][0]


        yield context_embedded, context_len, features, question_embedded, masked_question, starts, ends
        
        
def get_batch(pivot, data, meta, coords, train_flag=True):
    
    data_size = len(data)
    shuffled_indexes = np.arange(0, data_size, 1)
    np.random.shuffle(shuffled_indexes)
    shuffled_indexes = shuffled_indexes[:pivot]
    
    context_embedded = np.zeros((pivot, C_MAX_SIZE, EMBEDDING_SIZE))
    context_len = np.zeros((pivot), dtype=np.int32)
    features = np.zeros((pivot, C_MAX_SIZE, FEATURES_SIZE), dtype=np.float64)
    
    question_embedded = np.zeros((pivot, Q_MAX_SIZE, EMBEDDING_SIZE))
    masked_question = np.zeros((pivot), dtype=np.int32)
    
    starts = np.zeros((pivot))
    ends = np.zeros((pivot))
    
    for i in range(pivot):
        
        question = data[shuffled_indexes[i]][5] 
        masked_question[i] = len(question)
         
        for j in range(len(question)):
            question_embedded[i][j] = meta['embedding'][question[j]]
        for j in range(len(question), Q_MAX_SIZE):
            question_embedded[i][j] = meta['embedding'][0]
        
        if train_flag:
            starts[i] = data[shuffled_indexes[i]][8]
            ends[i] = data[shuffled_indexes[i]][9]
        else:
            starts[i] = coords[shuffled_indexes[i]][0]
            ends[i] = coords[shuffled_indexes[i]][1]
            
        paragraph = data[shuffled_indexes[i]][1]
        
        context_len[i] = len(paragraph)
        for j in range(len(paragraph)):
            context_embedded[i][j] = meta['embedding'][paragraph[j]]
            features[i][j][:len(data[shuffled_indexes[i]][2][j])] = data[shuffled_indexes[i]][2][j]
            features[i][j][len(data[shuffled_indexes[i]][2][j]) + data[shuffled_indexes[i]][4][j]] = 1
            features[i][j][len(data[shuffled_indexes[i]][2][j]) + ent_num + data[shuffled_indexes[i]][3][j]] = 1
        for j in range(len(paragraph), C_MAX_SIZE):
            context_embedded[i][j] = meta['embedding'][0]
        
        
    return context_embedded, context_len, features, question_embedded, masked_question, starts, ends


def get_batch_demo(context, question, meta, word2index):
    pivot = BATCH_SIZE
    
    context_embedded = np.zeros((pivot, C_MAX_SIZE, EMBEDDING_SIZE))
    context_len = np.zeros((pivot), dtype=np.int32)
    features = np.zeros((pivot, C_MAX_SIZE, FEATURES_SIZE), dtype=np.float64)
   
    question_embedded = np.zeros((pivot, Q_MAX_SIZE, EMBEDDING_SIZE))
    masked_question = np.zeros((pivot), dtype=np.int32)
 
    count = 0
    # на случай если такого слова нет, надо ????
    tokenized_context = tokenize(context)
    print("tokenized context")
    print(tokenized_context)
   
    nums_context = [word2index.get(word, 1) for word in tokenized_context]
 
    context_len[0] = len(nums_context)
   
    for j in range(len(nums_context)):
        context_embedded[0][j] = meta['embedding'][nums_context[j]]
       
    for j in range(len(nums_context), C_MAX_SIZE):
        context_embedded[0][j] = meta['embedding'][0]
 
    nums_question = [word2index.get(word, 1) for word in tokenize(question)]
    masked_question[0] = len(nums_question)
 
    for j in range(len(nums_question)):
        question_embedded[0][j] = meta['embedding'][nums_question[j]]
    for j in range(len(nums_question), Q_MAX_SIZE):
        question_embedded[0][j] = meta['embedding'][0]
       
    return context_embedded, context_len, features, question_embedded, masked_question



# def generator(contexts_input=contexts_input,
#               contexts_len=contexts_len,
#               questions_input=questions_input,
#               questions_len=questions_len,
#               ans_start=ans_start,
#               ans_end=ans_end,
#               batch_size=32,
#               shuffle=True):
    
#     #хотим понять, сколько есть контекстов
#     size = len(contexts_input)
#     a = 0

#     if shuffle:
#         index = shuffled_index([len(c) for c in contexts_input])
#     else:
#         index = list(range(size))
     
#     for start in range(0, size, batch_size):
#         end = min(start + batch_size, size)
#         if (end < start + batch_size):
#             break
            

#         idxs = index[start: end]
#         #print(idxs)
        
#         # теперь надо сделать так чтобы все инпуты были одинакового размера
        
#         arg1 = embedder(simple_pad(contexts_input[start:end], max_par_size))
#         arg2 = embedder(simple_pad(questions_input[start:end], max_que_size))

#         yield [

#             #simple_pad(contexts_input[start:end], max_par_size),
#             #embedder(simple_pad(contexts_input[start:end], max_par_size)),
#             arg1,
#             arg2,
    
#             #simple_pad(questions_input[start:end], max_que_size),
#             #embedder(simple_pad(questions_input[start:end], max_que_size)),
            
#             contexts_len[start:end],
#             questions_len[start:end],
            
#             ans_start[start:end],
#             ans_end[start:end] 
#         ]
     
    
# def generator_1(contexts_input=contexts_input,
#               contexts_len=contexts_len,
#               questions_input=questions_input,
#               questions_len=questions_len,
#               ans_start=ans_start,
#               ans_end=ans_end,
#               batch_size=32,
#               shuffle=True):
    
#     #хотим понять, сколько есть контекстов
#     size = len(contexts_input)
#     a = 0

#     if shuffle:
#         index = shuffled_index([len(c) for c in contexts_input])
#     else:
#         index = list(range(size))
     
#     for start in range(0, size, batch_size):
#         end = min(start + batch_size, size)
#         if (end < start + batch_size):
#             break
            

#         idxs = index[start: end]
#         #print(idxs)
        
#         # теперь надо сделать так чтобы все инпуты были одинакового размера
#         yield [

#             embedder(simple_pad(contexts_input[start:end], max_par_size),
#                      features_1[start:end], 
#                      tag_ids[start:end], 
#                      ent_ids[start:end],
#                      extra=True),
#             embedder(simple_pad(questions_input[start:end], max_que_size)),

#             contexts_len[start:end],
#             questions_len[start:end],
            
#             ans_start[start:end],
#             ans_end[start:end] 
#         ]

     
    
### для скачивания предподготовленных данных фейсбука

def get_fb_data_data(download=False):
    PATH='data.msgpack'
    if download:
        wget.download('https://www.dropbox.com/s/r33ljlagys0wscb/data.msgpack?dl=1', out=PATH)
    with open(PATH, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    return data

def get_fb_data_meta(download=False):
    PATH='meta.msgpack'
    if download:
        wget.download('https://www.dropbox.com/s/83txkgiqmdlv1m3/meta.msgpack?dl=1', out=PATH)
    with open(PATH, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    return data

# def start_finder(context, answer, tokens):
#     index_start = context.find(answer)
#     start = len(tokens)
#     for i in range(len(tokens)):
#         if tokens[i][0] == index_start:
#             start = i
#             break
#     return start

# def end_finder(start, context, answer, tokens):
#     index_start = context.find(answer)
#     prob_end = index_start + len(answer)
#     end = len(tokens)
    
#     for i in range(start, len(tokens)):
#         if tokens[i][1] == prob_end:
#             end = i
#             break
            
#     return end + 1

def find_answers_in_dev(dev):
    def find1(text, answer, tokens):
        look_index = text.find(answer)
        beg = len(tokens)
        for i in range(len(tokens)):
            if tokens[i][0] == look_index:
                beg = i
                break
        return beg
    
    def find2(beg, text, ans, tokens):
        index = text.find(ans)
        prob_end = index + len(ans)
        end = len(tokens)
        for i in range(beg, len(tokens)):
            if tokens[i][1] == prob_end:
                end = i
                break
        return end + 1
    
    coordinates = np.zeros((len(dev), 2))

    for i in range(len(dev)):
        beg = find1(dev[i][6], dev[i][8][0], dev[i][7])
        coordinates[i][0] = beg
        coordinates[i][1] = find2(beg, dev[i][6], dev[i][8][0], dev[i][7])
    return coordinates


def F_temp(beg_p, end_p, beg_r, end_r):
    TP = max((min(end_r, end_p) - max(beg_p, beg_r)), 0)
    FP = max((max(end_r, end_p) - min(beg_p, beg_r) - TP), 0)
     
    if FP == 0 and TP == 0:
        return 0
    return ((2 * TP) / (2 * TP + FP))


def F_score(beg_predicted, end_predicted, beg_real, end_real, size):
    result = 0
    for i in range(size):
        beg_r = beg_real[i]
        end_r = end_real[i]
        beg_p = np.argmax(beg_predicred[i])
        end_p = np.argmax(end_predicted[i])
        
        result += F_temp(beg_p, end_p, beg_r, end_r)
    return result / size
        
    
def main():
    print("prepare script checking: DONE")
    print("all functions are ok")
    return
    

if __name__ == '__main__':
    main()

