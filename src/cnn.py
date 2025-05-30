import numpy as np
import pickle
from typing import Tuple, List, Dict, Any
import tensorflow as tf
from tensorflow import keras

class ConvLayer:
    def __init__(self, weights: np.ndarray, bias: np.ndarray, strides: Tuple[int, int] = (1, 1), padding: str = 'valid', activation: str = 'relu'):
        self.weights = weights  
        self.bias = bias        
        self.strides = strides
        self.padding = padding
        self.activation = activation
        
    def apply_activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'linear':
            return x
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, input_height, input_width, input_channels = x.shape
        filter_height, filter_width, _, num_filters = self.weights.shape
        
        if self.padding == 'valid':
            output_height = (input_height - filter_height) // self.strides[0] + 1
            output_width = (input_width - filter_width) // self.strides[1] + 1
            padded_x = x
        else:  
            output_height = int(np.ceil(input_height / self.strides[0]))
            output_width = int(np.ceil(input_width / self.strides[1]))
            
            pad_along_height = max((output_height - 1) * self.strides[0] + filter_height - input_height, 0)
            pad_along_width = max((output_width - 1) * self.strides[1] + filter_width - input_width, 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            padded_x = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        output = np.zeros((batch_size, output_height, output_width, num_filters))

        for b in range(batch_size):
            for f in range(num_filters):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.strides[0]
                        h_end = h_start + filter_height
                        w_start = w * self.strides[1]
                        w_end = w_start + filter_width
                        region = padded_x[b, h_start:h_end, w_start:w_end, :]                        
                        output[b, h, w, f] = np.sum(region * self.weights[:, :, :, f]) + self.bias[f]
        return self.apply_activation(output)

class PoolingLayer:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] = (2, 2), pool_type: str = 'max'):
        self.pool_size = pool_size
        self.strides = strides
        self.pool_type = pool_type
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, input_height, input_width, channels = x.shape
        pool_height, pool_width = self.pool_size
        
        output_height = (input_height - pool_height) // self.strides[0] + 1
        output_width = (input_width - pool_width) // self.strides[1] + 1
        output = np.zeros((batch_size, output_height, output_width, channels))
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.strides[0]
                        h_end = h_start + pool_height
                        w_start = w * self.strides[1]
                        w_end = w_start + pool_width
                        region = x[b, h_start:h_end, w_start:w_end, c]
                        if self.pool_type == 'max':
                            output[b, h, w, c] = np.max(region)
                        elif self.pool_type == 'average':
                            output[b, h, w, c] = np.mean(region)
        return output

class FlattenLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

class GlobalPoolingLayer:
    def __init__(self, pool_type: str = 'average'):
        self.pool_type = pool_type
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.pool_type == 'average':
            return np.mean(x, axis=(1, 2))
        elif self.pool_type == 'max':
            return np.max(x, axis=(1, 2))

class DenseLayer:
    def __init__(self, weights: np.ndarray, bias: np.ndarray, activation: str = 'relu'):
        self.weights = weights  
        self.bias = bias        
        self.activation = activation
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

class CNNFromScratch:
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        """Add a layer to the network"""
        self.layers.append(layer)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = self.forward(x)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

def load_keras_model_weights(keras_model_path: str) -> Dict[str, Any]:
    model = keras.models.load_model(keras_model_path)
    weights_dict = {}
    
    layer_idx = 0
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            layer_name = layer.__class__.__name__.lower()
            
            if 'conv' in layer_name:
                weights_dict[f'conv_{layer_idx}'] = {
                    'weights': weights[0],  
                    'bias': weights[1],     
                    'strides': layer.strides,
                    'padding': layer.padding
                }
            elif 'dense' in layer_name:
                activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'linear'
                weights_dict[f'dense_{layer_idx}'] = {
                    'weights': weights[0],  
                    'bias': weights[1],     
                    'activation': activation
                }
            elif 'pooling' in layer_name or 'pool' in layer_name:
                pool_type = 'max' if 'max' in layer_name else 'average'
                weights_dict[f'pool_{layer_idx}'] = {
                    'pool_size': layer.pool_size,
                    'strides': layer.strides,
                    'pool_type': pool_type
                }
            
            layer_idx += 1
    return weights_dict

def build_cnn_from_keras(keras_model_path: str) -> CNNFromScratch:
    keras_model = keras.models.load_model(keras_model_path)
    cnn = CNNFromScratch()
    for layer in keras_model.layers:
        layer_name = layer.__class__.__name__.lower()
        if 'conv' in layer_name and hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            
            activation = 'linear'
            if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
                activation = layer.activation.__name__
            
            conv_layer = ConvLayer(
                weights=weights[0],
                bias=weights[1],
                strides=layer.strides,
                padding=layer.padding,
                activation=activation
            )
            cnn.add_layer(conv_layer)
        elif 'activation' in layer_name:
            continue
        elif 'maxpool' in layer_name or 'averagepool' in layer_name:
            pool_type = 'max' if 'max' in layer_name else 'average'
            pool_layer = PoolingLayer(
                pool_size=layer.pool_size,
                strides=layer.strides,
                pool_type=pool_type
            )
            cnn.add_layer(pool_layer)
        elif 'flatten' in layer_name:
            flatten_layer = FlattenLayer()
            cnn.add_layer(flatten_layer)
        elif 'globalpooling' in layer_name or 'globalaverage' in layer_name:
            pool_type = 'max' if 'max' in layer_name else 'average'
            global_pool_layer = GlobalPoolingLayer(pool_type=pool_type)
            cnn.add_layer(global_pool_layer)
        elif 'dense' in layer_name and hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()
            activation = 'linear'
            if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
                activation = layer.activation.__name__
            dense_layer = DenseLayer(
                weights=weights[0],
                bias=weights[1],
                activation=activation
            )
            cnn.add_layer(dense_layer)
    return cnn

def save_model_weights(model: keras.Model, filepath: str):
    model.save(filepath)
    print(f"Model saved to {filepath}")

def calculate_macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> float:
    f1_scores = []
    for class_idx in range(num_classes):
        true_positives = np.sum((y_true == class_idx) & (y_pred == class_idx))
        false_positives = np.sum((y_true != class_idx) & (y_pred == class_idx))
        false_negatives = np.sum((y_true == class_idx) & (y_pred != class_idx))
        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives/(true_positives + false_positives)
            
        if true_positives + false_negatives == 0:
            recall = 0
        else:
            recall= true_positives / (true_positives + false_negatives)
            
        if precision + recall == 0:
            f1 = 0
        else:
            f1=2 * (precision * recall)/ (precision + recall)
        f1_scores.append(f1)
    return np.mean(f1_scores)

def test_implementation_consistency(keras_model_path: str, test_data: Tuple[np.ndarray, np.ndarray], 
                                 num_samples: int = 100) -> Dict[str, float]:
    keras_model = keras.models.load_model(keras_model_path)
    scratch_model = build_cnn_from_keras(keras_model_path)
    x_test, y_test = test_data
    x_sample = x_test[:num_samples]
    y_sample = y_test[:num_samples]
    
    keras_pred = keras_model.predict(x_sample)
    keras_pred_classes = np.argmax(keras_pred, axis=1)
    scratch_pred_classes = scratch_model.predict(x_sample)
    keras_f1 = calculate_macro_f1_score(y_sample, keras_pred_classes)
    scratch_f1 = calculate_macro_f1_score(y_sample, scratch_pred_classes)

    implementation_accuracy = np.mean(keras_pred_classes == scratch_pred_classes)
    return {
        'keras_f1': keras_f1,
        'scratch_f1': scratch_f1,
        'implementation_accuracy':implementation_accuracy,
        'mean_absolute_error': np.mean(np.abs(keras_pred - scratch_model.predict_proba(x_sample)))
    }