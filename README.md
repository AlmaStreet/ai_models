# ai_models

This project goes through creating AI models from scratch using TensorFlow, PyTorch, and Hugging Face libraries.

# TensorFlow and PyTorch

Setting up virtual environment
```
python3 -m venv .venv_tf_pt
source .venv_tf_pt/bin/activate
```

## TensorFlow
###  Part 1: Basic Model
Create TensorFlow model with 2 features
```
python3 main_tf.py
```

Run TensorFlow tests
```
python3 test_main_tf.py
```
### Part 2: Multi-Class Classification and Fine-Tuning
#### Multi-Class Classification Model
Input: 4 features (fur length, tail type, ear shape, speed)<br>
Output: Classifies between cat, dog, fox, rabbit<br>
Created from scratch (not fine-tuned)
```
python3 main_tf_categorical.py
```

Test model
```
python3 test_main_tf_categorical.py
```
#### Fine-Tuning by Adding Layers
Takes the pre-trained 2-feature model and adds extra layers
Expands output categories to cat, dog, fox, rabbit
```
python3 main_tf_finetune_add_layers.py
```

Test model
```
python3 test_main_tf_finetune_add_layers.py
```
#### Full Fine-Tuning (Re-training All Layers)
Takes the original model and unfreezes all layers for full training
Retains 2 input features but modifies layers to improve performance
```
python3 main_tf_finetune_full.py
```

Test model
```
python3 test_main_tf_finetune_full.py
```


## PyTorch
Create PyTorch model
```
python3 main_pt.py
```

Run TensorFlow tests
```
python3 test_main_pt.py
```


# Hugging Face Part 1

Setting up virtual environment
```
python3 -m venv .venv_hf
source .venv_hf/bin/activate
```

Create Hugging Face model
```
python3 main_hf.py
```

Run Hugging Face tests
```
python3 test_main_hf.py
```
