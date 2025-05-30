import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
from datasets import load_dataset

class RNNLayer:
    def __init__(self, units, return_sequences=False, bidirectional=False):
        self.units = units
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.initialized = False
        
    def initialize(self, input_shape):
        input_dim = input_shape[-1]
        limit = np.sqrt(6 / (input_dim + self.units))
        
        self.W_xh = np.random.uniform(-limit, limit, (input_dim, self.units))
        self.W_hh = np.random.uniform(-limit, limit, (self.units, self.units))
        self.b_h = np.zeros((self.units,))
        
        if self.bidirectional:
            self.W_xh_back = np.random.uniform(-limit, limit, (input_dim, self.units))
            self.W_hh_back = np.random.uniform(-limit, limit, (self.units, self.units))
            self.b_h_back = np.zeros((self.units,))
        
        self.initialized = True
    
    def forward_step(self, x, h_prev, W_xh, W_hh, b_h):
        h = np.tanh(x @ W_xh + h_prev @ W_hh + b_h)
        return h
    
    def forward(self, x):
        if not self.initialized:
            self.initialize(x.shape)
        
        batch_size, seq_len, input_dim = x.shape
        h_forward = np.zeros((batch_size, self.units))
        
        if self.return_sequences:
            outputs = np.zeros((batch_size, seq_len, self.units))
        else:
            outputs = np.zeros((batch_size, self.units))
        
        for t in range(seq_len):
            h_forward = self.forward_step(x[:, t, :], h_forward, self.W_xh, self.W_hh, self.b_h)
            if self.return_sequences:
                outputs[:, t, :] = h_forward
        if not self.return_sequences:
            outputs = h_forward
        if self.bidirectional:
            h_backward = np.zeros((batch_size, self.units))
            if self.return_sequences:
                outputs_back = np.zeros((batch_size, seq_len, self.units))
            else:
                outputs_back = np.zeros((batch_size, self.units))
        
            for t in range(seq_len-1, -1, -1):
                h_backward = self.forward_step(x[:, t, :], h_backward, self.W_xh_back, self.W_hh_back, self.b_h_back)
                if self.return_sequences:
                    outputs_back[:, t, :] = h_backward
            
            if not self.return_sequences:
                outputs_back = h_backward
            
            if self.return_sequences:
                outputs = np.concatenate((outputs, outputs_back), axis=-1)
            else:
                outputs = np.concatenate((outputs, outputs_back), axis=-1)
        
        return outputs

class EmbeddingLayer:
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
    
    def forward(self, x):
        return self.embedding_matrix[x]

class DropoutLayer:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
    
    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
            return x * self.mask
        return x

class DenseLayer:
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.initialized = False
    
    def initialize(self, input_shape):
        input_dim = input_shape[-1]
        limit = np.sqrt(6 / (input_dim + self.units))
        self.W = np.random.uniform(-limit, limit, (input_dim, self.units))
        self.b = np.zeros((self.units,))
        self.initialized = True
    
    def forward(self, x):
        if not self.initialized:
            self.initialize(x.shape)
        
        z = x @ self.W + self.b
        
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        else:
            return z

class RNNModelFromScratch:
    def __init__(self, keras_model):
        self.layers = []
        self.build_from_keras_model(keras_model)
    
    def build_from_keras_model(self, keras_model):
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Embedding):
                embedding_matrix = layer.get_weights()[0]
                self.layers.append(EmbeddingLayer(embedding_matrix))
            
            elif isinstance(layer, (tf.keras.layers.SimpleRNN, tf.keras.layers.LSTM, tf.keras.layers.GRU)):
                units = layer.units
                return_sequences = layer.return_sequences
                bidirectional = isinstance(layer, tf.keras.layers.Bidirectional)
                
                rnn_layer = RNNLayer(units, return_sequences, bidirectional)
                if bidirectional:
                    forward_layer = layer.forward_layer
                    backward_layer = layer.backward_layer
                    
                    kernel, recurrent_kernel, bias = forward_layer.get_weights()
                    rnn_layer.W_xh = kernel
                    rnn_layer.W_hh = recurrent_kernel
                    rnn_layer.b_h = bias
                    
                    kernel, recurrent_kernel, bias = backward_layer.get_weights()
                    rnn_layer.W_xh_back = kernel
                    rnn_layer.W_hh_back = recurrent_kernel
                    rnn_layer.b_h_back = bias
                else:
                    kernel, recurrent_kernel, bias = layer.get_weights()
                    rnn_layer.W_xh = kernel
                    rnn_layer.W_hh = recurrent_kernel
                    rnn_layer.b_h = bias
                
                rnn_layer.initialized = True
                self.layers.append(rnn_layer)
            
            elif isinstance(layer, tf.keras.layers.Dropout):
                self.layers.append(DropoutLayer(layer.rate))
            
            elif isinstance(layer, tf.keras.layers.Dense):
                dense_layer = DenseLayer(layer.units, activation=layer.activation.__name__ if layer.activation else None)
                
                kernel, bias = layer.get_weights()
                dense_layer.W = kernel
                dense_layer.b = bias
                dense_layer.initialized = True
                
                self.layers.append(dense_layer)
    
    def forward(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
        return x
    
    def predict(self, x):
        logits = self.forward(x, training=False)
        if logits.shape[-1] > 1:  
            return np.argmax(logits, axis=-1)
        else:  
            return (logits > 0).astype(int)

def load_and_prepare_data():
    dataset = load_dataset("indonlp/NusaX-sentiment", "ind")
    
    
    texts = [example['text'] for example in dataset['train']]
    labels = [example['label'] for example in dataset['train']]
    
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42)
    
    return (train_texts, train_labels), (test_texts, test_labels)

def train_keras_model(train_texts, train_labels, test_texts, test_labels, rnn_layers_config, bidirectional=False, dropout_rate=0.2):
    vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=64)
    vectorizer.adapt(train_texts)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorizer)
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=128))
    
    for i, units in enumerate(rnn_layers_config):
        return_sequences = i < len(rnn_layers_config) - 1  
        rnn_layer = tf.keras.layers.SimpleRNN(units, return_sequences=return_sequences)
        if bidirectional:
            rnn_layer = tf.keras.layers.Bidirectional(rnn_layer)
        model.add(rnn_layer)
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(3,activation='softmax'))  
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(np.array(train_texts), np.array(train_labels),validation_data=(np.array(test_texts), np.array(test_labels)),epochs=15, batch_size=64)

    return model, history

def evaluate_model(model, texts, labels):
    predictions = model.predict(np.array(texts))
    predictions = np.argmax(predictions, axis=1)
    return f1_score(labels, predictions, average='macro')

def compare_implementations(keras_model, scratch_model, test_texts, test_labels):
    vectorizer = keras_model.layers[1]
    test_tokens = vectorizer(np.array(test_texts)).numpy()
    keras_probs = keras_model.predict(np.array(test_texts))
    keras_preds = np.argmax(keras_probs, axis=1)
    keras_f1 = f1_score(test_labels, keras_preds, average='macro')
    
    scratch_probs = scratch_model.forward(test_tokens, training=False)
    scratch_preds = np.argmax(scratch_probs, axis=1)
    scratch_f1 = f1_score(test_labels, scratch_preds, average='macro')
    
    print(f"Keras Model F1-score: {keras_f1:.4f}")
    print(f"Scratch Model F1-score: {scratch_f1:.4f}")
    print(f"Difference: {abs(keras_f1 - scratch_f1):.4f}")
    return keras_f1, scratch_f1