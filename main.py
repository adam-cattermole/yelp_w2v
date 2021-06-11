from skorch.callbacks import EpochScoring, Checkpoint, TensorBoard, TrainEndCheckpoint, Freezer
from skorch import NeuralNet
import skorch.dataset
from sklearn.metrics import roc_auc_score, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_validate
import gensim.downloader 
import torch
import torch.nn as nn
import torch.nn.functional as F
from spacecutter.models import OrdinalLogisticModel
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.callbacks import AscensionCallback
import os
import csv

def read_yelp(file):
     with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        return zip(*[(row[1], int(row[0])) for row in reader])


class YelpDataset(torch.utils.data.IterableDataset):
    """Load the train / validation / test json files
    
    Keyword Arguments:
        data_path: string the folder path where the Yelp data lives
        train_val: string {train, test}
    """
    def __init__(self, data_path, train_val):
        self.data_path = data_path
        self.train_val = train_val
        self.file_path = os.path.join(self.data_path, f"{self.train_val}.csv")
        self.text, self.label = read_yelp(self.file_path)
        
    def __len__(self):
        return len(self.text)
        
    def __getitem__(self, index):
        return self.text[index], self.label[index]


class DenseClassifier(nn.Module):
    """
    Dense model using an embedding with pretrained weights

    takes the mean of each word embedding in each sequence
    """
    def __init__(self, embedding, embedding_size, hidden_size, num_classes, input_length, dropout):
        super(DenseClassifier, self).__init__()
        self.embedding = embedding
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(input_length * embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_size, num_classes)

    @classmethod
    def from_pretrained(cls, weights, hidden_size, num_classes, input_length, dropout):
        embedding = nn.Embedding.from_pretrained(weights)
        embedding_size = weights.size()[1]
        return cls(embedding, embedding_size, hidden_size, num_classes, input_length, dropout)

    @classmethod
    def from_vocab(cls, vocab_size, embedding_size, hidden_size, num_classes, input_length, dropout):
        embedding = nn.Embedding(vocab_size, embedding_size)
        return cls(embedding, embedding_size, hidden_size, num_classes, input_length, dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return self.lin2(x)

import numpy as np

def numericalize_text(document, word_to_index, pad_examples=True, input_length=100):
    """Numericalise a single document and truncate/pad to a set length
    
    Keyword Arguments:
        document: a document
        word_to_index: a dictionary of word -> index
        pad_examples: boolean enable to pad or truncate examples to set input_length
        input_length: int (Optional)

    Returns:
    """
    if pad_examples:
        output = pad([word_to_index.get(word) for word in document if word in word_to_index], input_length)
    else:
        output = [word_to_index.get(word) for word in document if word in word_to_index]
    return output


def numericalize_corpus(corpus, word_to_index, pad_examples=True, input_length=100):
    """Numericalize a full corpus using a vocab

    Keyword arguments
        corpus: list[list[str]] a collection of documents
        word_to_index: a dictionary of word -> index
        pad_examples: boolean enable to pad or truncate examples to set input_length
        input_length: int the length to truncate / pad each document
    """
    return [numericalize_text(document, word_to_index, pad_examples, input_length) for document in corpus]


def pad(document, input_length):
    """Pad or truncate a document to a given length
    
    Keyword Arguments:
        document: list[int] a list containing the numericalized document
        input_length: int the length to truncate or pad the review to
    """
    length = len(document)
    if input_length > length:
        out_review = document + ([0] * (input_length - length))
    else:
        out_review = document[:input_length]
    return out_review


def encode_pre_trained_w2v(dataset, word_to_index, pad_examples=True, input_length=100):
    """Encode a dataset using pre-trained w2v embeddings
    
    Keyword Arguments:
        dataset: a pytorch dataset containing tuples of (text, label)
        word_to_index: a dictionary of mappings from words to indices (created using word_to_index fn.)

    Returns:
        A tuple of numpy vectors containing the encoded text and the associated labels
    """
    X = np.array(numericalize_corpus(dataset.text[:100], word_to_index, pad_examples, input_length))
    y = np.array(dataset.label[:100])

    return X, y

HIDDEN_SIZE = 32
INPUT_LENGTH = 100
DATA_PATH = os.getenv("DATA_PATH", "/datadrive/yelp/yelp_review_full_csv")
DEVICE="CPU"

# Load the gensim weights
google_news = gensim.downloader.load("word2vec-google-news-300")
weights = torch.FloatTensor(google_news.vectors)
word2index = {w: i for i, w in enumerate(google_news.index_to_key)}

# load the yelp data
training = YelpDataset(DATA_PATH, "train")

# Encode using w2v
X, y = encode_pre_trained_w2v(training, word2index, input_length=INPUT_LENGTH)

# Scale labels between 0 - 4 
y -= y.min()

classifier = DenseClassifier.from_pretrained(
    weights=weights,
    hidden_size=HIDDEN_SIZE,
    input_length=INPUT_LENGTH,
    num_classes=1,
    dropout=0.2)

for name, param in classifier.named_parameters():
  print(name)

print("define freezer")
freezer = Freezer("embedding.weight")

print("define net")
# define skorch model
net = NeuralNet(
    module=OrdinalLogisticModel,
    module__predictor=classifier,
    module__num_classes=5,
    optimizer=torch.optim.SGD, callbacks=[freezer, ('ascension', AscensionCallback())], # ensure that the cutpoints remain in the correct order
    criterion=CumulativeLinkLoss,
    train_split=skorch.dataset.CVSplit(2, stratified=False),
    iterator_train__shuffle=True,
    max_epochs=5,
    device="cpu"
)

print("fit net")
# Learn the weights
net.fit(X, torch.tensor(y, dtype=torch.int64).unsqueeze(-1))

print("read test data")
# read in test data
test = YelpDataset(DATA_PATH, "test")
X_test, y_test = encode_pre_trained_w2v(test, word2index, input_length=INPUT_LENGTH)

print("rescale labels")
# rescale the labels between 0 - 4
y_test -= y_test.min()

print("calc roc and acc")
# Calculate the roc and accuracy test data 
preds = net.predict_proba(X_test)
def mae_scorer(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred.argmax(axis=1))
def roc_scorer(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovo", labels=[0,1,2,3,4])

print("get mae and roc")
mae, roc = mae_scorer(y_test, preds), roc_scorer(y_test, preds)
print(f"mae: {mae}, roc: {roc}")

# from sklearn.metrics import roc_auc_score, mean_absolute_error, make_scorer
# from sklearn.model_selection import cross_validate, StratifiedKFold

# def mae_scorer(y_true, y_pred):
#     return mean_absolute_error(y_true, y_pred.argmax(axis=1))

# mae_scoring = make_scorer(mae_scorer,
#                           greater_is_better=False,
#                           needs_proba=True)

# def roc_scorer(y_true, y_pred):
#   return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovo", labels=[0,1,2,3,4])

# roc_scoring = make_scorer(roc_scorer, 
#                           greater_is_better=True, 
#                           needs_proba=True)

# kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
# scores_dense = cross_validate(
#             net, X, y=torch.tensor(y, dtype=torch.float).unsqueeze(-1), 
#             cv=kfolds, scoring={"ROC": roc_scoring, "MAE": mae_scoring})

# scores_dense
