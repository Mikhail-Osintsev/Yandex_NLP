import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Вычисляем s^T * W_mult
    s_T_W = np.dot(decoder_hidden_state.T, W_mult)  # shape (1, n_features_enc)
    
    # Вычисляем attention scores
    attention_scores = np.dot(s_T_W, encoder_hidden_states)  # shape (1, n_states)
    
    # Вычисляем веса с помощью softmax
    weights = softmax(attention_scores)
    
    # Вычисляем итоговый attention vector
    attention_vector = np.dot(weights, encoder_hidden_states.T).T  # shape (n_features_enc, 1)
    
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Вычисляем W_add_enc * h_i
    W_enc_H = np.dot(W_add_enc, encoder_hidden_states)  # shape (n_features_int, n_states)
    
    # Вычисляем W_add_dec * s
    W_dec_s = np.dot(W_add_dec, decoder_hidden_state)  # shape (n_features_int, 1)
    
    # Суммируем с учётом broadcasting
    additive = W_enc_H + W_dec_s  # shape (n_features_int, n_states)
    
    # Применяем функцию активации tanh
    z = np.tanh(additive)  # shape (n_features_int, n_states)
    
    # Вычисляем attention scores
    attention_scores = np.dot(v_add.T, z)  # shape (1, n_states)
    
    # Вычисляем веса с помощью softmax
    weights = softmax(attention_scores)
    
    # Вычисляем итоговый attention vector
    attention_vector = np.dot(weights, encoder_hidden_states.T).T  # shape (n_features_enc, 1)
    
    return attention_vector