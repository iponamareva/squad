from prepare import *
from constants import *

import tensorflow as tf
import msgpack
import json
import re
import numpy as np
import collections
import random
import gensim

from nltk.tokenize import word_tokenize
from tqdm import tqdm_notebook as tqdm
from nltk.tokenize import RegexpTokenizer

def main():
    sess = tf.Session() 
    
    import_path = SAVE_MODEL_PATH + ".meta"
    #saver = tf.train.import_meta_graph(TEST_MODEL_PATH)
    saver = tf.train.import_meta_graph(import_path)
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    
    print("MODEL RESTORED FROM PATH: " + import_path)

    graph = tf.get_default_graph()
    
    #embedded_cotexts, contexts_len, embedded_features, embedded_question, masked_question, answer_starts, answer_ends, keep_prob, result_starts, result_ends = extract_graph(graph)
    
    embedded_contexts = graph.get_tensor_by_name("embedded_contexts:0")
    contexts_len =  graph.get_tensor_by_name("contexts_len:0")
    embedded_features = graph.get_tensor_by_name("embedded_features:0")
        
    embedded_question = graph.get_tensor_by_name("embedded_question:0")
    masked_question = graph.get_tensor_by_name("masked_question:0")
    
    answer_starts = graph.get_tensor_by_name("answer_starts:0")
    answer_ends = graph.get_tensor_by_name("answer_ends:0")
    
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    
    
    result_starts = graph.get_tensor_by_name("result_starts:0")
    result_ends = graph.get_tensor_by_name("result_ends:0")
    
    print("GRAPH LOADED")
    
    data = get_fb_data_data()
    meta = get_fb_data_meta()
    
    coords = find_answers_in_dev(data['dev'])
    
    print("TEST DATA LOADED")
    
    print("RUNNING ITERATION ON TEST")
    
    (arg1, arg2, arg3, arg4, arg5, arg6, arg7) = get_batch(BATCH_SIZE, data['dev'], meta, coords, False)

    b_vector_test, e_vector_test = sess.run([result_starts, result_ends],
        feed_dict = {
            embedded_contexts: arg1,
            contexts_len: arg2,
            embedded_features: arg3,

            embedded_question: arg4,
            masked_question: arg5,
            answer_starts: arg6,
            answer_ends: arg7,

            keep_prob: 1.0,

        }
    )

    F_score_TEST = F_score(b_vector_test, e_vector_test, arg6, arg7, BATCH_SIZE)
    print("TEST F SCORE:   " + str(F_score_TEST))
    print("-"*80)

    print("TESTING on DEV COMPLETED")
    
    
if __name__ == '__main__':
    main()
