# -*- coding: utf-8 -*-
"""Kopie von homework_5.ipynb

The following assignment consists of a theoretical part (learning portfolio) and a practical part (assignment). The goal is to build a classification model that predicts from which subject area a certain abstract originates. The plan would be that next week we will discuss your learnings from the theory part, that means you are relatively free to fill your Learning Portfolio on this new topic and in two weeks we will discuss your solutions of the Classification Model.

#Theory part (filling your Learning Portfolio, May 10)

In preparation for the practical part, I ask you to familiarize yourself with the following resources in the next week:

1) Please watch the following video:

https://course.fast.ai/Lessons/lesson4.html

You are also welcome to watch the accompanying Kaggle notebook if you like the video.

2) In addition to the video, I recommend you to read the first chapters of the course

https://huggingface.co/learn/nlp-course/chapter1/1


Try to understand principle processes and log them in your learning portfolio! A few suggestions: What is a pre-trained NLP model? How do I load them? What is tokenization? What does fine-tuning mean? What types of NLP Models are there? What possibilities do I have with the Transformers package? etc...

#Practical part (Assignment, May 17)

1) Preprocessing: The data which I provide as zip in Olat must be processed first, that means we need a table which has the following form:

Keywords | Title | Abstract | Research Field

The research field is determined by the name of the file.

2) We need a training dataset and a test dataset. My suggestion would be that for each research field we use the first 5700 lines for the training dataset and the last 300 lines for the test dataset. Please stick to this because then we can compare our models better!

3) Please use a pre-trained model from huggingface to build a classification model that tries to predict the correct research field from the 26. Please calculate the accuracy and the overall accuracy for all research fields. If you solve this task in a group, you can also try different pre-trained models. In addition to the abstracts, you can also see if the model improves if you include keywords and titles.

Some links, which can help you:

https://huggingface.co/docs/transformers/training

https://huggingface.co/docs/transformers/tasks/sequence_classification

One last request: Please always use PyTorch and not TensorFlow!
"""

!pip install transformers==4.28.0 accelerate datasets evaluate

import pandas as pd
import glob 
import os 
from google.colab import drive
from google.colab import data_table
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

data_table.enable_dataframe_formatter()

drive.mount('/content/drive')

dir = '/content/drive/MyDrive/Colab Notebooks/data 2/'

'''had to skip bad lines due to errors when reading in'''

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for filename in os.listdir(dir):
    if filename.endswith('.csv'): 
        file_path = os.path.join(dir, filename)
        data = pd.read_csv(file_path, on_bad_lines="skip")

        research_field = filename.split('_')[0]   
        data['Research Field'] = research_field
        data['Keywords'] = data['Author Keywords'].fillna('') + ' ' + data['Index Keywords'].fillna('')

        columns_to_keep = ['Keywords', 'Research Field', 'Abstract', 'Title']
        data = data[columns_to_keep]

        # Replace empty abstracts with title and keywords
        data.loc[data['Abstract'] == '[No abstract available]', 'Abstract'] = data['Title'] + ' ' + data['Keywords']

        # Split into Training and Test
        train_end_idx = int(len(data) * 0.95)
        train_data = pd.concat([train_data, data[:train_end_idx]])
        test_data = pd.concat([test_data, data[train_end_idx:]])

# Split into training and validation using stratified sampling
train_data, validation_data = train_test_split(train_data, test_size=0.20, stratify=train_data['Research Field'], random_state=42)

print("Length of training, validation and test set")
print((len(train_data), len(validation_data), len(test_data)))

validation_data.head(10)

from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
from datasets import Dataset,DatasetDict

#rename columns and create datasets
def prepare_dataset(df):
    df = df.loc[:, ['Abstract', 'Research Field']]
    df.columns = ['text', 'labels']
    return Dataset.from_pandas(df)

# Load data and create datasets
train_ds, valid_ds, test_ds = [prepare_dataset(df) for df in [train_data, validation_data, test_data]]
dataset_dict = DatasetDict({'train': train_ds, 'validation': valid_ds, 'test': test_ds})

#use model (a little similar to bert)
model = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model)

'''define the labels of the tokenized dataset'''

id2label = {1: 'DENT', 2: 'AGRI', 3: 'ENER', 4: 'PSYC', 5: 'DECI', 6: 'VETE', 7: 'PHAR', 8: 'MATH',
       9: 'NURS', 10: 'ECON', 11: 'COMP', 12: 'ARTS', 13: 'CENG', 14: 'ENVI', 15: 'SOCI', 16: 'BIOC',
       17: 'MATE', 18: 'CHEM', 19: 'HEAL', 20: 'ENGI', 21: 'BUSI', 22: 'NEUR', 23: 'MEDI', 24: 'IMMU',
       25: 'PHYS', 0: 'EART'}
label2id = {value: key for key, value in id2label.items()}

# Define function to tokenize text and labels
def tokenize_function(x):
    tokens = tokenizer(x['text'], truncation=True, padding="max_length")
    tokens["labels"] = [label2id[label] for label in x["labels"]]
    return tokens

# Tokenize datasets
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

from transformers import BertForSequenceClassification
import evaluate

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("/content/drive/MyDrive/Colab Notebooks/data 2/checkpoints/", evaluation_strategy="epoch", per_device_train_batch_size=8, per_device_eval_batch_size=8)
model = BertForSequenceClassification.from_pretrained(model, num_labels=26, id2label=id2label, label2id=label2id)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=True)

from collections import defaultdict, Counter

correct_single_hits = 0
corect_highest_three_categories = 0
incorrect_single_hits_per_category = defaultdict(int)
incorrect_triple_hits_per_category = defaultdict(int)

for i in range(len(predictions.predictions)):
  if np.argmax(preds.predictions[i]) == tokenized_datasets["test"][i]["labels"]:
    correct_single_hits += 1
  if tokenized_datasets["test"][i]["labels"] in np.argpartition(preds.predictions[i], -3)[-3:]:
    corect_highest_three_categories += 1
  else:
    incorrect_single_hits_per_category[id2label[tokenized_datasets["test"][i]["labels"]]] += 1
    incorrect_triple_hits_per_category[id2label[tokenized_datasets["test"][i]["labels"]]] += 1

sorted_single_hit_fails = dict(Counter(incorrect_single_hits_per_category).most_common())
sorted_triple_hit_fails = dict(Counter(incorrect_triple_hits_per_category).most_common())

print("Single and Triple Hit Accuracy")
print((correct_single_hits/len(preds.predictions), corect_highest_three_categories/len(preds.predictions)))
print("Category to wrong prediction single")
print(sorted_single_hit_fails)
print("Category to wrong prediction triple")
print(sorted_triple_hit_fails)

"""Addition: Accuracy measures whether the research field with the highest probability value matches the target. With 26 research fields, it would also be interesting to know if the correct target is at least among the three highest probability values.

$\begin{pmatrix} A\\ B \\ C \\D \\E \end{pmatrix} = \begin{pmatrix} 0.1\\ 0.95 \\ 0.5 \\0.2 \\0.3 \end{pmatrix} → \text{Choice}_1 = B, \text{Choice}_3 = B,C,E$
"""
