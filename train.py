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

from prepare import *
from constants import *


def main():
    
    data = get_fb_data_data(DOWNLOAD)
    train = data['train']
    meta = get_fb_data_meta(DOWNLOAD)
    
    ### собственно сама модель и её обучение

    tf.reset_default_graph()
    
    embedded_contexts = tf.placeholder(dtype=tf.float32, name="embedded_contexts", shape=(BATCH_SIZE, C_MAX_SIZE, EMBEDDING_SIZE))
    contexts_len = tf.placeholder(dtype=tf.int32, name="contexts_len", shape=(BATCH_SIZE))
    
    embedded_features = tf.placeholder(dtype=tf.float32, name="embedded_features", shape=(BATCH_SIZE, C_MAX_SIZE, FEATURES_SIZE))
    
    embedded_question = tf.placeholder(dtype=tf.float32, name="embedded_question", shape=(BATCH_SIZE, Q_MAX_SIZE, EMBEDDING_SIZE))
    masked_question = tf.placeholder(dtype=tf.int32, name="masked_question", shape=(BATCH_SIZE))
    
    answer_starts = tf.placeholder(dtype=tf.int32, name="answer_starts", shape=(BATCH_SIZE))
    answer_ends = tf.placeholder(dtype=tf.int32, name="answer_ends", shape=(BATCH_SIZE))

    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob", shape=())
    
    print("PLACEHOLDERS: DONE")
    print("BUILDING GRAPH")


    
    
    # Описание всех LSTM клеток, которые будут использоваться в дальнейшем
    # Первые 4 относятся к вопросу
    # Вторые 4 относятся к контексту
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, keep_prob)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_2, keep_prob)
    lstm_cell_3 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    lstm_cell_3 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_3, keep_prob)
    lstm_cell_4 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    lstm_cell_4 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_4, keep_prob)
    lstm_cell_5 = tf.nn.rnn_cell.BasicLSTMCell(2 * HIDDEN_SIZE)
    lstm_cell_5 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_5, keep_prob)
    lstm_cell_6 = tf.nn.rnn_cell.BasicLSTMCell(2 * HIDDEN_SIZE)
    lstm_cell_6 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_6, keep_prob)
    lstm_cell_7 = tf.nn.rnn_cell.BasicLSTMCell(2 * HIDDEN_SIZE)
    lstm_cell_7 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_7, keep_prob)
    lstm_cell_8 = tf.nn.rnn_cell.BasicLSTMCell(2 * HIDDEN_SIZE)
    lstm_cell_8 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_8, keep_prob)
    
    
    #  Вопрос - два слоя бидирекшонал, внимание

    question_outputs, question_state = tf.nn.bidirectional_dynamic_rnn(lstm_cell_1, lstm_cell_2,inputs=embedded_question, sequence_length=masked_question, dtype=tf.float32)
    question_input = tf.concat([question_outputs[0], question_outputs[1]], -1)

    question_outputs, question_state = tf.nn.bidirectional_dynamic_rnn(lstm_cell_3, lstm_cell_4,inputs=question_input,sequence_length=masked_question, dtype=tf.float32, scope="SCOPE__1")

    question_output = tf.concat(question_outputs, -1)
    
    
    attention = tf.layers.dense(inputs=question_output,units=1,use_bias=True)
    attention = tf.reshape(attention, (BATCH_SIZE, Q_MAX_SIZE))

    question_mask = tf.sequence_mask(masked_question, maxlen=Q_MAX_SIZE, dtype=tf.float32)
    
    attention = tf.multiply(tf.nn.softmax(attention), question_mask)
    attention = tf.reshape(attention, (BATCH_SIZE, Q_MAX_SIZE, 1))

    question_output = tf.matmul(tf.transpose(question_output, perm=[0, 2, 1]), attention)

    question_output = tf.reshape(question_output, (BATCH_SIZE, 2 * HIDDEN_SIZE))
    q_state = tf.contrib.rnn.LSTMStateTuple(question_output, question_output)

    # Контекст - два слоя бидирекшонал
    embedded_inputs = tf.concat([embedded_contexts, embedded_features], axis = -1)
    context_outputs, context_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_5, initial_state_fw = q_state, cell_bw=lstm_cell_6, initial_state_bw = q_state, sequence_length=contexts_len,inputs=embedded_inputs,dtype=tf.float32, scope="SCOPE_2")

    context_input = tf.concat([context_outputs[0], context_outputs[1]], -1)
    
    context_outputs_, context_state_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_7, cell_bw=lstm_cell_8, initial_state_fw = q_state, initial_state_bw = q_state, inputs=context_input, dtype=tf.float32, scope="SCOPE_1")

    context_outputs = tf.concat(context_outputs_, -1)
    
    result_starts = tf.layers.dense(inputs=context_outputs, units=2*HIDDEN_SIZE, use_bias=True)
    result_ends = tf.layers.dense(inputs=context_outputs, units=2*HIDDEN_SIZE, use_bias=True)
    result_starts = tf.matmul(result_starts, tf.reshape(question_output, (BATCH_SIZE, 2*HIDDEN_SIZE, 1)), name="result_starts")
    result_ends = tf.matmul(result_ends, tf.reshape(question_output, (BATCH_SIZE, 2*HIDDEN_SIZE, 1)), name="result_ends")

    loss_starts = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_starts, logits=tf.reshape(result_starts, (BATCH_SIZE, C_MAX_SIZE)), name="result_starts_out"))
    
    
    loss_ends = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_ends, logits=tf.reshape(result_ends, (BATCH_SIZE, C_MAX_SIZE)), name="result_end_out"
        )
    )

    total_loss = loss_starts + loss_ends

    optimizer = tf.train.RMSPropOptimizer(0.001)
    gradients, variables = zip(*optimizer.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1000)
    train_step = optimizer.apply_gradients(zip(gradients, variables))

    coords = find_answers_in_dev(data['dev'])
    
    #  Обучение и сохранение модели
    
    print("GRAPH IS BUILT")

    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    print("STARTING LEARNING")
    print("THE FOLLOWING LOGS WILL SHOW YOU HOW LEARNING PROCEEDS")
    print("*"*80)
    
    if EP_FLAG:
        for epochs in range(EPOCHS):
            i = -1
            print("EPOCHE: " + str(epochs + 1))
            
            for (arg1_c, arg2_cl, arg3_f, arg4_q, arg5_qm, arg6_s, arg7_e) in batch_gen(BATCH_SIZE, train, meta, None):
                i = i + 1

                loss_result, _, begins, ends = sess.run([total_loss, train_step, result_starts, result_ends], 
                                             feed_dict={
                                                 embedded_contexts: arg1_c,
                                                 contexts_len: arg2_cl,
                                                 embedded_features: arg3_f,
                                                 embedded_question: arg4_q,
                                                 masked_question: arg5_qm,
                                                 answer_starts: arg6_s,
                                                 answer_ends: arg7_e,
                                                 keep_prob: 0.7,
                                                 })


                if i % CHECKER_STEP == 0:
                    F_score_TRAIN = F_score(begins, ends, arg6_s, arg7_e, BATCH_SIZE)
                    print("iteration " + str(i) + " LOSS: " + str(loss_result) + "\nTRAIN F SCORE : " + str(F_score_TRAIN))

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

                if i % SAVER_STEP == 0 and i != 0:
                    saver.save(sess, SAVE_STEP_MODEL_PATH, global_step=i)
    
    if EP_FLAG=False:
        for i in (range(ITERATIONS)):
            (arg1_c, arg2_cl, arg3_f, arg4_q, arg5_qm, arg6_s, arg7_e) = get_batch(BATCH_SIZE, train, meta, None)

            loss_result, _, begins, ends = sess.run([total_loss, train_step, result_starts, result_ends], 
                                         feed_dict={
                                             embedded_contexts: arg1_c,
                                             contexts_len: arg2_cl,
                                             embedded_features: arg3_f,
                                             embedded_question: arg4_q,
                                             masked_question: arg5_qm,
                                             answer_starts: arg6_s,
                                             answer_ends: arg7_e,
                                             keep_prob: 0.7,
                                             })


            if i % CHECKER_STEP == 0:
                F_score_TRAIN = F_score(begins, ends, arg6_s, arg7_e, BATCH_SIZE)
                print("iteration " + str(i) + " LOSS: " + str(loss_result) + "\nTRAIN F SCORE : " + str(F_score_TRAIN))

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

            if i % SAVER_STEP == 0 and i != 0:
                saver.save(sess, SAVE_STEP_MODEL_PATH, global_step=i)
            
    saver.save(sess, SAVE_MODEL_PATH)

    print("LEARNING IS DONE")
    print("SAVED GRAPH")

    

if __name__ == '__main__':
    main()
