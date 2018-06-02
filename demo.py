import msgpack
import json
import re
import numpy as np
import collections
import random
import gensim
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm

from constants import *
from prepare import *

def func(sess, embedded_contexts, contexts_len, embedded_features, embedded_question, keep_prob, masked_question, result_starts, result_ends):
     
    meta = get_fb_data_meta()
    word2index = {w: i for i, w in enumerate(meta['vocab'])}
    
    print("ДЛЯ ВЫХОДА нажмите q")
    stop_flag=False
    while True:
        id_ = 0
        try:
            while True:
                context = input('Контекст для вопроса ')
                if (context == "q"):
                    stop_flag=True
                    print("FINISHING")
                    break
                   
                if context.strip():
                    break                
                   
            if stop_flag:
                break
               
            while True:
                question = input('Вопрос ')
                if question.strip():
                    break
        except EOFError:
            break
 
        (arg1, arg2, arg3, arg4, arg5) = get_batch_demo(context, question, meta, word2index)
 
        b_vector, e_vector = sess.run(
                [result_starts,
                 result_ends],
                feed_dict = {
                    embedded_contexts: arg1,
                    contexts_len: arg2,
                    embedded_features: arg3,
                    embedded_question: arg4,
                    masked_question: arg5,
                    keep_prob: 1.0
                }
            )
   
        b = np.argmax(b_vector[0].reshape((-1, )))
        e = np.argmax(e_vector[0].reshape((-1, )))
       
        tock_context = tokenize(context)
        current_cont_len = len(tock_context)
       
        print(current_cont_len)
        print(b)
        print(e)
       
        if (e >= current_cont_len):
            print("Something went wrong with end - optimizing answer")
            e = current_cont_len - 1
        if (b >= current_cont_len):
            print("Something went wrong with begin - optimizing answer")
            b = current_cont_len - 1
        print("**********Ваш ответ*********")
        print(normalize(tock_context[b:e + 1]))
        
        
def main():
    tf.reset_default_graph()
    sess=tf.Session()
    import_path = USE_DEMO_MODEL_PATH + ".meta"
    saver = tf.train.import_meta_graph(import_path)
    saver.restore(sess,tf.train.latest_checkpoint('./'))
 
    graph = tf.get_default_graph()
    embedded_contexts = graph.get_tensor_by_name("embedded_contexts:0")
    contexts_len =  graph.get_tensor_by_name("contexts_len:0")
    embedded_features = graph.get_tensor_by_name("embedded_features:0")
    embedded_question = graph.get_tensor_by_name("embedded_question:0")
    masked_question = graph.get_tensor_by_name("masked_question:0")
    result_starts = graph.get_tensor_by_name("result_starts:0")
    result_ends = graph.get_tensor_by_name("result_ends:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
 
    func(sess, embedded_contexts, contexts_len, embedded_features, embedded_question, keep_prob, masked_question, result_starts, result_ends)
 
 
if __name__ == '__main__':
    main()
