import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score
import pickle
import json

class LSTMFromScratch:
    def __init__(self):
        self.weights = {}
        self.biases = {}
        self.embedding_weights = None
        self.vocab_size = None
        self.embedding_dim = None
        self.sequence_length = None
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def load_keras_model(self, model_path):
        self.keras_model = keras.models.load_model(model_path)
    
        for i, layer in enumerate(self.keras_model.layers):
            layer_name = f"layer_{i}_{layer.name}"
            
            if isinstance(layer, layers.Embedding):
                self.embedding_weights = layer.get_weights()[0]
                self.vocab_size, self.embedding_dim = self.embedding_weights.shape
                
            elif isinstance(layer, (layers.LSTM, layers.Bidirectional)):
                if isinstance(layer, layers.Bidirectional):
                
                    forward_weights = layer.forward_layer.get_weights()
                    backward_weights = layer.backward_layer.get_weights()
                    self.weights[f"{layer_name}_forward"] = forward_weights
                    self.weights[f"{layer_name}_backward"] = backward_weights
                else:
                
                    self.weights[layer_name] = layer.get_weights()
                    
            elif isinstance(layer, layers.Dense):
                w, b = layer.get_weights()
                self.weights[f"{layer_name}_w"] = w
                self.weights[f"{layer_name}_b"] = b
    
    def embedding_forward(self, input_ids):
        embedded = self.embedding_weights[input_ids]
        return embedded
    
    def lstm_cell_forward(self, x_t, h_prev, c_prev, weights):
        W_i, W_f, W_c, W_o = weights[0][:, :self.embedding_dim], weights[0][:, self.embedding_dim:2*self.embedding_dim], \
                              weights[0][:, 2*self.embedding_dim:3*self.embedding_dim], weights[0][:, 3*self.embedding_dim:]
        U_i, U_f, U_c, U_o = weights[1][:, :weights[1].shape[1]//4], weights[1][:, weights[1].shape[1]//4:weights[1].shape[1]//2], \
                              weights[1][:, weights[1].shape[1]//2:3*weights[1].shape[1]//4], weights[1][:, 3*weights[1].shape[1]//4:]
        b_i, b_f, b_c, b_o = weights[2][:weights[2].shape[0]//4], weights[2][weights[2].shape[0]//4:weights[2].shape[0]//2], \
                              weights[2][weights[2].shape[0]//2:3*weights[2].shape[0]//4], weights[2][3*weights[2].shape[0]//4:]
        
        i_t = self.sigmoid(np.dot(x_t, W_i.T) + np.dot(h_prev, U_i.T) + b_i)
        f_t = self.sigmoid(np.dot(x_t, W_f.T) + np.dot(h_prev, U_f.T) + b_f)
        c_tilde = self.tanh(np.dot(x_t, W_c.T) + np.dot(h_prev, U_c.T) + b_c)
        o_t = self.sigmoid(np.dot(x_t, W_o.T) + np.dot(h_prev, U_o.T) + b_o)
    
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * self.tanh(c_t)
        return h_t, c_t
    
    def lstm_forward(self, x, weights, return_sequences=False):
        batch_size, seq_len, input_dim = x.shape
        hidden_dim = weights[1].shape[0]
        
        h = np.zeros((batch_size, hidden_dim))
        c = np.zeros((batch_size, hidden_dim))
        outputs = []
        
        for t in range(seq_len):
            h, c = self.lstm_cell_forward(x[:, t, :], h, c, weights)
            outputs.append(h.copy())
        
        if return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]
    
    def bidirectional_lstm_forward(self, x, forward_weights, backward_weights, return_sequences=False):
        forward_output = self.lstm_forward(x, forward_weights, return_sequences=True)
        x_reversed = x[:, ::-1, :]
        backward_output = self.lstm_forward(x_reversed, backward_weights, return_sequences=True)
        backward_output = backward_output[:, ::-1, :] 
        
        combined_output = np.concatenate([forward_output, backward_output], axis=-1)
        
        if return_sequences:
            return combined_output
        else:
            return combined_output[:, -1, :]
    
    def dense_forward(self, x, weights_key):
        w = self.weights[f"{weights_key}_w"]
        b = self.weights[f"{weights_key}_b"]
        return np.dot(x, w) + b
    
    def forward(self, input_ids):
        x = self.embedding_forward(input_ids)
        
        for i, layer in enumerate(self.keras_model.layers[1:], 1): 
            layer_name = f"layer_{i}_{layer.name}"
            
            if isinstance(layer, layers.Bidirectional):
                x = self.bidirectional_lstm_forward(
                    x, 
                    self.weights[f"{layer_name}_forward"],
                    self.weights[f"{layer_name}_backward"],
                    return_sequences=False
                )
            elif isinstance(layer, layers.LSTM):
                x = self.lstm_forward(x, self.weights[layer_name], return_sequences=False)
            elif isinstance(layer, layers.Dropout):
            
                continue
            elif isinstance(layer, layers.Dense):
                x = self.dense_forward(x, layer_name)
                if layer == self.keras_model.layers[-1]: 
                    x = self.softmax(x)
        
        return x

def create_lstm_model(embedding_dim=100, lstm_units=64, num_lstm_layers=1, 
                     bidirectional=False, vocab_size=10000, num_classes=3):
    model = keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, mask_zero=True))
    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1) 
        if bidirectional:
            model.add(layers.Bidirectional(
                layers.LSTM(lstm_units, return_sequences=return_sequences, dropout=0.3)
            ))
        else:
            model.add(layers.LSTM(lstm_units, return_sequences=return_sequences, dropout=0.3))
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    f1 = f1_score(y_test, y_pred_classes, average='macro')
    return f1, y_pred_classes

def compare_implementations(keras_model, lstm_scratch, X_test, y_test):
    keras_pred = keras_model.predict(X_test)
    keras_pred_classes = np.argmax(keras_pred, axis=1)
    keras_f1 = f1_score(y_test, keras_pred_classes, average='macro')

    scratch_pred = lstm_scratch.forward(X_test)
    scratch_pred_classes = np.argmax(scratch_pred, axis=1)
    scratch_f1 = f1_score(y_test, scratch_pred_classes, average='macro')
    
    print(f"Keras F1 Score: {keras_f1:.4f}")
    print(f"From-scratch F1 Score: {scratch_f1:.4f}")
    print(f"Difference: {abs(keras_f1 - scratch_f1):.4f}")
    
    return keras_f1, scratch_f1