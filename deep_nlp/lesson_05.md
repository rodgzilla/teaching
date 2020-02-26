# Lesson 5

## Transfer learning

Content mostly taken from PyTorch tutorial on [transfer
learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

To load a pretrained ImageNet classification model, we can simply use
the torchvision module as follow

```python
from torchvision import models

model_conv = models.resnet18(pretrained = True)
print(model_conv)
```

By printing a model, we recursively list all of its components. It is
important to note that, except when in `nn.Sequential` block, we do
not know how these layers are used during the forward pass of the
model.

```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

You should by now be familiar with most of the layer types used in
this architecture.

We can notice that in the `layerX` subnetworks of this model, the
downsampling operation (usually performed by a maximum pooling
operation) is done using *strided convolution*:

```python
(downsample): Sequential(
  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```

![Strided convolution](../figures/no_padding_strides_convolution.gif)

(Once again, animation from
[vdumoulin](https://github.com/vdumoulin/conv_arithmetic))

Keep this detail in mind as it will be used during the practical work.

We then freeze all the layers of the model to set the finetuning
procedure up.

```python
for param in model_conv.parameters():
    param.requires_grad = False
```

By doing this, we say to the optimizer that we do not want these
parameters to be modified during the training procedure.

As the classification task we are dealing with has two classes, we
have to change the last linear layer of the model to output two
values. To properly size it, we first fetch the `in_features`
parameter of the current last layer.

```python
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(
    in_features  = num_ftrs,
    out_features = 2
)
```

We then setup the rest of the training as usual.

```python
model_conv       = model_conv.to(device)
criterion        = nn.CrossEntropyLoss()
optimizer_conv   = optim.SGD(model_conv.fc.parameters(), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

The training loop stays the same as usual.

```python
train_model(model_ft, device, criterion, optimizer_ft, exp_lr_scheduler,
            num_epochs=25)
```

By using this transfer learning logic, we are able to solve a lot of
different computer vision problem.

This PyTorch tutorial is well written and I would strongly encourage
the students to complete it and try to apply it to their own problem
or Kaggle competitions such as [Dogs vs. Cats
Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition). By
applying methods presented in tutorials to new problems, one generally
learns a tons of new information.

Recently, a small revolution has been happening in the machine
learning world with the discovery of very powerful transfer learning
method for Natural Language Processing (NLP) tasks. These advances are
mainly explained in the following research articles [Universal
Language Model Fine-tuning for Text
Classification](https://arxiv.org/abs/1801.06146), [Improving Language
Understanding by Generative
Pre-Training](https://openai.com/blog/language-unsupervised/) and
[BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding](https://arxiv.org/abs/1810.04805). Before taking a look
at these articles, we have to study the fundamentals of applying deep
learning to NLP tasks.

## NLP methods

### Embedding

Words are not continuous values which is the type of inputs neural
networks accept. In order to apply deep learning algorithms to textual
data, we have to transform the sequences of words into numerical
tensors. To do this, we will associate to each word of the vocabulary
a *vector* that will encode various information relevant to the task
we are solving, the vector corresponding to a word is called its
[*Embedding*](https://en.wikipedia.org/wiki/Word_embedding).

An [embedding
layer](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding) is
simply a lookup table of the embeddings. It is a tensor of shape
`[number of word in vocabulary, embedding dimension]`

```python
>>> embedding_layer = nn.Embedding(num_embeddings = 5, embedding_dim = 2)
>>> embedding_layer.weight
Parameter containing:
tensor([[-0.2029,  0.9372],
        [-0.5283, -0.6736],
        [ 1.4233,  0.7148],
        [ 0.2876,  0.6953],
        [ 1.1457,  1.0806]], requires_grad=True)

>>> embedding_layer.weight.shape
torch.Size([5, 2])

>>> word_index_sequence = torch.tensor([[0, 1, 0, 2]])
>>> word_index_sequence.shape
torch.Size([1, 4])

>>> embedded_sequence = embedding_layer(word_index_sequence)
>>> embedded_sequence
tensor([[[-0.2029,  0.9372],
         [-0.5283, -0.6736],
         [-0.2029,  0.9372],
         [ 1.4233,  0.7148]]], grad_fn=<EmbeddingBackward>)

>>> embedded_sequence.shape
torch.Size([1, 4, 2])
```

The embedding vectors of each word of the vocabulary are initialized
randomly. As you can notice, the weights of the embedding layer have
`requires_grad = True` which mean that they will be trained during the
whole model training procedure.

The vectors associated to each of the words will evolve during the
training to *encode* useful information about the words relative to
the task at hand. The word embeddings learned on a specific task may
be completely different to the ones trained on another unrelated task
even though they encode the same words.

Historically, when deep learning practitioners wanted to use neural
networks to solve NLP tasks, they used pre-trained embeddings. Using
these general purpose embeddings was the NLP version of *transfer
learning*.

There exist various tasks that have been used in the literature, for
example the Continuous Bag-of-Words model.

![Continuous Bag-of-Words](../figures/cbow.png)

In this model, we train a model to predict the word in the middle of a
window using the embeddings of its surrounding words as input. This
task encourages the model to create word embeddings that encode
information about their *context*. The goal is to exploit the idea
that "You shall know a word by the company it keeps" by [John Rupert
Firth](https://en.wikipedia.org/wiki/John_Rupert_Firth).

Another similar method is the Continuous Skip-gram model.

![Continuous Skip-gram](../figures/skip_gram.png)

In this model, we use the embedding word as input to a model trying to
predict its *context*.

After the training procedure, we can analyze the word embeddings that
we obtain to get a sense of the information and relations between
concepts that they encode.

![Country and Capital relationship in
embeddings](../figures/embedding_property.png)

In the previous figure, we choose a set of words corresponding to
countries and their respective capitals. We then select from our
embedding matrix the set of vectors corresponding to these words. As
these vectors have 1000 dimensions, we reduce their dimensionality by
computing linear combinations of their coordinates using a [Principal
component
analysis](https://en.wikipedia.org/wiki/Principal_component_analysis). The
type of relation appears without additional supervision. The model has
learned the concept of country as `China` and `Russia` often appear in
the same *contexts*, the concept of capital as `Berlin` and `Paris`
often appear in the same contexts.

When trained on a sufficiently big corpus, word embeddings contained a
lot of different information on the words they are associated to.

![Skip-gram relationships](../figures/embedding_relationships.png)

Pretrained word embeddings matrices include
[word2vec](https://en.wikipedia.org/wiki/Word2vec) and
[GloVe](https://nlp.stanford.edu/projects/glove/).

The source of these figures and more information on these methods can
be found in the articles in which they were defined: [Efficient
Estimation of Word Representations in Vector
Space](https://arxiv.org/abs/1301.3781) and [Distributed
Representations of Words and Phrases and their
Compositionality](https://arxiv.org/abs/1310.4546).

Much more information on word level semantics learning is available in
the Oxford Deep NLP course at [Lecture 2a- Word Level
Semantics](https://github.com/oxford-cs-deepnlp-2017/lectures#3-lecture-2a--word-level-semantics-ed-grefenstette)
([here for the lecture
video](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_2a_lexical_semantics.mp4))
and in the Christopher Olah's blog post [Deep Learning, NLP, and
Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/).

### Text preprocessing

Now that we *theorically* know how to transform a sequence of words
into a valid neural network input, let's see how to do it in practice.

In this section and the following ones, we will use a small sentiment
classification dataset from University of California Irvine. The raw
version of the dataset is available
[here](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)
and a preprocessed version is available in this repository
[here](../datasets/comments.txt).

To load this dataset, we will use
[pandas](https://pandas.pydata.org/), a very popular data manipulation
tool. Getting in-depth knowledge of pandas is not part of the course,
more information on how to use it are available in the "Introduction
to Data Science Course" in this repository.

```python
import pandas as pd

df            = pd.read_csv('comments.txt', sep = '\t', names = ['comment', 'label'])
df['comment'] = df.comment.str.lower()

for _, row in df.head().iterrows():
  print(f'Label {row.label}, Comment "{row.comment}"')
```

```
Label 0, Comment "so there is no way for me to plug it in here in the us unless i go by a converter."
Label 1, Comment "good case, excellent value."
Label 1, Comment "great for the jawbone."
Label 0, Comment "tied to charger for conversations lasting more than 45 minutes.major problems!!"
Label 1, Comment "the mic is great."
```

Next, we need to split the comments into words and take a look at the
number of words repartition. The operation of splitting a text into a
sequence of words or *tokens* is called *tokenization*. It is usually
performed using much more advanced algorithms than simply cutting
whenever we see a space character. We use this method just to build a
minimal example.

```python
df['split_comment'] = df.comment.str.split()
df['n_tokens']      = df.split_comment.apply(len)
print(*df.head().split_comment, sep = '\n')
```

```
['so', 'there', 'is', 'no', 'way', 'for', 'me', 'to', 'plug', 'it', 'in', 'here', 'in', 'the', 'us', 'unless', 'i', 'go', 'by', 'a', 'converter.']
['good', 'case,', 'excellent', 'value.']
['great', 'for', 'the', 'jawbone.']
['tied', 'to', 'charger', 'for', 'conversations', 'lasting', 'more', 'than', '45', 'minutes.major', 'problems!!']
['the', 'mic', 'is', 'great.']
```

```python
print(*df.head().n_tokens, sep = ', ')
```

```
21, 4, 4, 11, 4
```

```python
df.n_tokens.hist(bins = 20)
```

![Token counts in comments](../figures/token_count_comments.png)

As we can see, there is a great variability in the number of tokens in
each comment. There exist some model architectures that are able to
deal with variable length inputs but we have not seen them yet. We are
going to preprocess the data so that all comments have the same number
of tokens.

```python
def pad_comment(token_list, final_len, pad_token = '<pad>'):
  if len(token_list) > final_len:
    return token_list[:final_len]
  return token_list + [pad_token] * (final_len - len(token_list))
```

This function truncates the token list if it is too long and add
`<pad>` until `final_len` is reached otherwise. Let's apply it to our
comments.

```python
from functools import partial

max_token_comment = df.n_tokens.max()
df['padded_comment'] = df.split_comment.apply(
    partial(
        pad_comment,
        final_len = max_token_comment if max_token_comment % 2 == 0 else (max_token_comment + 1)
    )
)

print(df.padded_comment.iloc[1])
```

```
['good', 'case,', 'excellent', 'value.', '<pad>', '<pad>', '<pad>',
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
 '<pad>', '<pad>', ..., ]
```
We pad the sequences to an even length for reasons we will see later.

Now that our sequences are preprocessed, we need to build our
*vocabulary*, the set of words that can appear in comments.

```python
vocabulary = set()
for comment in df.split_comment:
  vocabulary.update(set(comment))
print(*list(vocabulary)[:5], sep = '\n')
```

```
boiled
happy,
think
world,
perplexing.
```

Now that we have our vocabulary, we need to create two mappings: one
from tokens to vocabulary index and another one from vocabulary index
to tokens.

```python
token_to_idx = {
    '<oov>': 0,
    '<pad>': 1,
    **{
        token: (idx + 2)
        for idx, token in enumerate(vocabulary)
    }
}

print('Number of words in the vocabulary:', len(token_to_idx))
print(token_to_idx['<pad>'])
print(token_to_idx['the'])
print(token_to_idx['happy'])
```

```
Number of words in the vocabulary: 7352
1
6477
7176
```

You can notice that we added two special tokens to our vocabulary,
`<pad>` and `<oov>`. We have already what `<pad>` is used for. The
`<oov>` will be used at test time to indicate the presence of Out Of
Vocabulary (OOV) words.

To generate the second mapping, we simply reverse this one.

```python
idx_to_token = {
    idx: token
    for token, idx in token_to_idx.items()
}
print(idx_to_token[6477])
print(idx_to_token[1234])
```

```
the
wash
```

We will use these two mappings to encode comments that we want to feed
to the model and decode comments that have been given as input to the
model. Let's encode our comment dataset.

```python
def encode(comment_tokens, token_to_idx):
  return [token_to_idx.get(token, 0) for token in comment_tokens]

def decode(comment_token_indices, idx_to_token):
  try:
    first_pad_index = comment_token_indices.index(1)
  except:
    first_pad_index = len(comment_token_indices)
  return ' '.join(idx_to_token[token_id] for token_id in comment_token_indices[:first_pad_index])
```

Let's now test these functions.

```python
comment = df.padded_comment.iloc[3]
print(comment[:15])
encoded_comment = encode(comment, token_to_idx)
print(encoded_comment[:15])
print(decode(encoded_comment, idx_to_token))
```

```
['tied', 'to', 'charger', 'for', 'conversations', 'lasting', 'more', 'than', '45', 'minutes.major', 'problems!!', '<pad>', '<pad>', '<pad>', '<pad>']
[257, 5192, 313, 645, 4893, 6427, 4619, 3652, 3156, 4918, 6922, 1, 1, 1, 1]
tied to charger for conversations lasting more than 45 minutes.major problems!!
```

Now that we know that our functions work, let's encode the whole dataset.

```python
df['encoded_comment'] = df.padded_comment.apply(
    partial(
        encode,
        token_to_idx = token_to_idx
    )
)
```

Now that our comments are in a proper format, we can convert them into
PyTorch objects.

```python
X = torch.tensor(df.encoded_comment)
y = torch.tensor(df.label)
print(X.shape, y.shape)
dataset = TensorDataset(X, y)
```

```
torch.Size([3000, 72]) torch.Size([3000])
```

We have `3000` comments and each of them is a sequence of `72` tokens
(a lot of them being padding tokens). For each comment, the label `y`
tells us whether is is positive `1` or negative `0`.

To evaluate our model, we are going to split the dataset into two
parts, a *training set* that will be used to train it, and a *test
set* that will not be seen during training that will be used to
*evaluate* the model's *generalization capabilities*.

```python
test_prop = .1
test_size = int(len(dataset) * test_prop)
train_dataset, test_dataset = random_split(dataset, [len(dataset) - test_size, test_size])
len(train_dataset), len(test_dataset)
```

```
(2700, 300)
```

Finally, we create our dataloaders that we will use to iterate through
the two datasets.

```python
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader  = DataLoader(test_dataset, batch_size = 32, shuffle = True)
```

### Training and evaluation functions

Our problem is a simple classification task. As we have properly
plugged our data into the PyTorch framework, the training and
evaluation loop does not differ from what we have seen until now.

```python
def eval(model, loader):
  model.eval()
  correct_pred = 0
  total_pred   = 0
  with torch.no_grad():
    for X, y in loader:
      y_pred        = model(X)
      y_pred_class  = y_pred.argmax(dim = 1)
      correct_pred += (y == y_pred_class).sum().item()
      total_pred   += len(y)

  return correct_pred / total_pred

def train(model, epochs, optimizer, criterion, train_loader, test_loader):
  for epoch in range(epochs):
    model.train()
    for batch_id, (X, y) in enumerate(train_loader):
      optimizer.zero_grad()
      y_pred = model(X)
      loss   = criterion(y_pred, y)
      loss.backward()
      optimizer.step()
    if epoch % 5 == 0:
    print(f'[{epoch:4}] Train eval {100 * eval(model, train_loader):5.3f}%, Test eval {100 * eval(model, test_loader):5.3f}%')
```

### MLP for NLP

Now that our data are properly formatted, we can create our first
neural network. In this section, the neural network will be a
multilayer perceptron.

```python
class MLPCommentClassifier(nn.Module):
  def __init__(self, emb_dim, voc_size, seq_len):
    super(MLPCommentClassifier, self).__init__()
    self.emb      = nn.Embedding(voc_size, emb_dim)
    self.lin1     = nn.Linear(seq_len * emb_dim, 64)
    self.dropout1 = nn.Dropout(.7)
    self.lin2     = nn.Linear(64, 64)
    self.dropout2 = nn.Dropout(.4)
    self.lin3     = nn.Linear(64, 2)

  def forward(self, x):                 # [batch, 72]
    x = self.emb(x)                     # [batch, 72, emb_dim]
    x = torch.flatten(x, start_dim = 1) # [batch, 72 * emb_dim]
    x = self.lin1(x)                    # [batch, 64]
    x = self.dropout1(x)                # [batch, 64]
    x = F.relu(x)                       # [batch, 64]
    x = self.lin2(x)                    # [batch, 64]
    x = self.dropout2(x)                # [batch, 64]
    x = F.relu(x)                       # [batch, 64]
    x = self.lin3(x)                    # [batch, 2]
    x = torch.log_softmax(x, dim = 1)   # [batch, 2]

    return x
```

In the forward method, `self.emb(x)` will replace each token index by
its corresponding embedding vector. The shape is then `[batch size,
sequence length, embedding dimension]`. As we have seen before, linear
layers only work on 1D inputs, because of this, we flatten the
sequence of embedding vectors to obtain a 1D array with `x =
torch.flatten(x, start_dim = 1)`.

The rest of the model is a very usual fully connected network. Let's
now instantiate the model and train it.

```python
model = MLPCommentClassifier(
    emb_dim  = 5,
    voc_size = len(token_to_idx),
    seq_len  = train_dataset[0][0].shape[0]
)
print(model)
```

```
MLPCommentClassifier(
  (emb): Embedding(7352, 5)
  (lin1): Linear(in_features=360, out_features=64, bias=True)
  (dropout1): Dropout(p=0.7, inplace=False)
  (lin2): Linear(in_features=64, out_features=64, bias=True)
  (dropout2): Dropout(p=0.4, inplace=False)
  (lin3): Linear(in_features=64, out_features=2, bias=True)
)
```

```python
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()
train(model, 151, optimizer, criterion, train_loader, test_loader)
```

```
[   0] Train eval 49.852%, Test eval 52.333%
[   5] Train eval 50.074%, Test eval 52.333%
[  10] Train eval 50.630%, Test eval 47.667%
[  15] Train eval 50.630%, Test eval 48.000%
[  20] Train eval 52.296%, Test eval 52.333%
[  25] Train eval 52.741%, Test eval 53.000%
[  30] Train eval 53.704%, Test eval 53.000%
[  35] Train eval 53.556%, Test eval 52.000%
[  40] Train eval 56.000%, Test eval 50.333%
[  45] Train eval 59.037%, Test eval 49.333%
[  50] Train eval 61.741%, Test eval 49.667%
[  55] Train eval 64.185%, Test eval 53.667%
[  60] Train eval 68.704%, Test eval 59.000%
[  65] Train eval 73.037%, Test eval 59.333%
[  70] Train eval 75.444%, Test eval 59.667%
[  75] Train eval 80.259%, Test eval 60.333%
[  80] Train eval 82.593%, Test eval 58.333%
[  85] Train eval 84.926%, Test eval 61.667%
[  90] Train eval 87.963%, Test eval 60.000%
[  95] Train eval 88.963%, Test eval 59.333%
[ 100] Train eval 90.444%, Test eval 60.333%
[ 105] Train eval 91.556%, Test eval 61.667%
[ 110] Train eval 92.037%, Test eval 60.333%
[ 115] Train eval 93.148%, Test eval 59.667%
[ 120] Train eval 92.667%, Test eval 61.667%
[ 125] Train eval 93.889%, Test eval 61.333%
[ 130] Train eval 94.481%, Test eval 60.667%
[ 135] Train eval 94.667%, Test eval 62.667%
[ 140] Train eval 94.778%, Test eval 63.000%
[ 145] Train eval 95.741%, Test eval 61.333%
[ 150] Train eval 95.852%, Test eval 61.333%
```

We see that this model *overfits* the training data quite a lot, while
not giving very good performances (63% accuracy at most compared to a
model always answering 1 that would have 50% accuracy). The dataset
available for this task is very small (only 3000 comments) so we will
probably not be able to fix all the overfitting using
regularization. There is still room for improvement by using models
able to use the *spacial* nature of textual data, Convolutional neural
networks.

### 1D convolutions for NLP

When we worked on image data, we used 2D convolutional networks
because pictures are two dimensional inputs, each pixel have neighbors
in two directions, above, below and on its sides.

In the world of NLP, inputs are sequences of words. Each word only has
neighbors on its sides, it is a *1 dimensional signal*. To work with
this kind of signal, we use *1D convolutions*. A 1D convolution is
applied to a 1D signal and is parametrized by 1D kernels.

```python
class CNNCommentClassifier(nn.Module):
  def __init__(self, emb_dim, voc_size, seq_len):
    super(CNNCommentClassifier, self).__init__()
    self.emb      = nn.Embedding(voc_size, emb_dim)
    self.conv1    = nn.Conv1d(emb_dim, 32, 3, padding = 1)
    self.dropout1 = nn.Dropout(.7)
    self.conv2    = nn.Conv1d(32     , 32, 3, padding = 1)
    self.dropout2 = nn.Dropout(.6)
    self.conv3    = nn.Conv1d(32     , 64, 3, padding = 1)
    self.dropout3 = nn.Dropout(.5)
    self.conv4    = nn.Conv1d(64     , 64, 3, padding = 1)
    self.dropout4 = nn.Dropout(.3)
    self.lin1     = nn.Linear((seq_len // 4) * 64, 64)
    self.lin2     = nn.Linear(64, 2)

  def forward(self, x):                 # [batch, 72]
    x = self.emb(x)                     # [batch, 72, 5]
    x = torch.transpose(x, 1, 2)        # [batch, 5, 72]
    x = self.conv1(x)                   # [batch, 32, 72]
    x = self.dropout1(x)                # [batch, 32, 72]
    x = F.relu(x)                       # [batch, 32, 72]
    x = self.conv2(x)                   # [batch, 32, 72]
    x = self.dropout2(x)                # [batch, 32, 72]
    x = F.relu(x)                       # [batch, 32, 72]
    x = F.max_pool1d(x, 2)              # [batch, 32, 36]
    x = self.conv3(x)                   # [batch, 64, 36]
    x = self.dropout3(x)                # [batch, 64, 36]
    x = F.relu(x)                       # [batch, 64, 36]
    x = self.conv4(x)                   # [batch, 64, 36]
    x = self.dropout4(x)                # [batch, 64, 36]
    x = F.relu(x)                       # [batch, 64, 36]
    x = F.max_pool1d(x, 2)              # [batch, 64, 18]
    x = torch.flatten(x, start_dim = 1) # [batch, 64 * 18]
    x = self.lin1(x)                    # [batch, 64]
    x = F.relu(x)                       # [batch, 64]
    x = self.lin2(x)                    # [batch, 2]
    x = torch.log_softmax(x, dim = 1)   # [batch, 2]

    return x
```

When working with convolutions in PyTorch, we want inputs of the shape
`[batch, channel, signal dimensions]`. When we worked with 2D signals
this was `[batch, channel, height, width]` and now, with token
sequences, it is `[batch, channel, token index]`. When we want to
apply a convolution after an embedding layer, the *embedding dimensions*
play the role of *channels*.

As we have seen previously, PyTorch embeddings output are of the shape
`[batch, token index, embedding dimension]` as this is not what we
want, we have to
[*transpose*](https://en.wikipedia.org/wiki/Transpose) the last two
dimensions. We perform this operation using `torch.transpose(x, 1,
2)`.

Just as we use `nn.Conv1d` in place of `nn.Conv2d`, we use
`F.max_pool1d` in place of `F.max_pool2d`.

```python
model = CNNCommentClassifier(
    emb_dim = 5,
    voc_size = len(token_to_idx),
    seq_len = train_dataset[0][0].shape[0]
)
print(model)
```

```
CNNCommentClassifier(
  (emb): Embedding(7352, 5)
  (conv1): Conv1d(5, 32, kernel_size=(3,), stride=(1,), padding=(1,))
  (dropout1): Dropout(p=0.7, inplace=False)
  (conv2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,))
  (dropout2): Dropout(p=0.6, inplace=False)
  (conv3): Conv1d(32, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (dropout3): Dropout(p=0.5, inplace=False)
  (conv4): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (dropout4): Dropout(p=0.3, inplace=False)
  (lin1): Linear(in_features=1152, out_features=64, bias=True)
  (lin2): Linear(in_features=64, out_features=2, bias=True)
)
```

```python
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()
train(model, 151, optimizer, criterion, train_loader, test_loader)
```

```
[   0] Train eval 49.815%, Test eval 51.667%
[   5] Train eval 58.889%, Test eval 54.333%
[  10] Train eval 65.519%, Test eval 58.667%
[  15] Train eval 67.815%, Test eval 62.000%
[  20] Train eval 77.593%, Test eval 63.000%
[  25] Train eval 81.519%, Test eval 64.333%
[  30] Train eval 86.889%, Test eval 66.000%
[  35] Train eval 89.037%, Test eval 66.333%
[  40] Train eval 92.370%, Test eval 67.667%
[  45] Train eval 93.778%, Test eval 66.667%
[  50] Train eval 94.630%, Test eval 67.667%
[  55] Train eval 96.593%, Test eval 69.667%
[  60] Train eval 97.148%, Test eval 69.667%
[  65] Train eval 97.630%, Test eval 69.667%
[  70] Train eval 97.593%, Test eval 67.667%
[  75] Train eval 98.667%, Test eval 69.667%
[  80] Train eval 98.444%, Test eval 68.667%
[  85] Train eval 99.185%, Test eval 70.667%
[  90] Train eval 98.889%, Test eval 69.000%
[  95] Train eval 98.481%, Test eval 68.667%
[ 100] Train eval 99.556%, Test eval 70.667%
[ 105] Train eval 99.630%, Test eval 71.667%
[ 110] Train eval 99.222%, Test eval 68.333%
[ 115] Train eval 99.815%, Test eval 70.000%
[ 120] Train eval 99.778%, Test eval 70.333%
[ 125] Train eval 99.741%, Test eval 69.333%
[ 130] Train eval 99.593%, Test eval 69.667%
[ 135] Train eval 99.852%, Test eval 71.000%
[ 140] Train eval 99.741%, Test eval 68.667%
[ 145] Train eval 99.889%, Test eval 71.000%
[ 150] Train eval 99.889%, Test eval 71.333%
```

Once again, we see a big overfit but the accuracy on the test set is
much better.

### Recurrent neural network

*Recurrent neural networks* (RNNs) are another family of neural
network. They are specifically designed to work with sequences.

![RNN illustration](../figures/rnn.png)

(Image from [Chris Olah's
blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))

A recurrent neural network can be thought of as multiple copies of the
same network, each passing a message to a successor (citation from
Olah's blog). For each token of the sequence, the neural network is
applied and takes two inputs: the token itself and the hidden output
computed by the RNN on the previous token. You can take a look at
PyTorch
[tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
on recurrent neural network from scratch to get a sense of how they
work.

In the code of this section, we will use a Long Short-Term Memory
network (from [this
article](http://web.eecs.utk.edu/~ielhanan/courses/ECE-692/Bobby_paper1.pdf))
which is a more "robust" version of the simple RNN. A precise
explanation of its inner working is beyond the scope of this
course. The architecture of this model is described in the following
diagram.

![Long short-term memory architecture](../figures/lstm.png)

```python
class RNNCommentClassifier(nn.Module):
  def __init__(self, emb_dim, voc_size, seq_len):
    super(RNNCommentClassifier, self).__init__()
    self.emb = nn.Embedding(voc_size, emb_dim)
    self.rnn = nn.LSTM(
        input_size    = emb_dim,
        hidden_size   = 64,
        bidirectional = True, # We look at the sequence from start to end and from
                              # end to start at the same type. It is a typical "hack"
                              # of RNN to make them more powerful
        dropout       = .3
    )
    self.dropout = nn.Dropout(.2)
    # 2 directions
    self.lin     = nn.Linear(64 * 2, 2)

  def forward(self, x):                                    # [batch, 72]
    x                      = self.emb(x)                   # [batch, 72, 5]
    x                      = torch.transpose(x, 0, 1)      # [72, batch, 5]
    output, (hidden, cell) = self.rnn(x)                   # output = [72, batch, 64]
                                                           # hidden = [layers (1) * num directions (2) = 2, batch, 64]
                                                           # cell   = [layers (1) * num_directions (2) = 2, batch, 64]
    hidden                 = torch.transpose(hidden, 0, 1) # hidden = [batch, layers (1) * num_directions (2) = 2, 64]
    hidden                 = torch.cat((
        hidden[:, 0, :],
        hidden[:, 1, :]
        ),
        dim = -1
    )                                                      # hidden = [batch, hidden (64) * num_directions (2) = 128]
    hidden                 = self.dropout(hidden)          # [batch, 128]
    x                      = self.lin(hidden)              # [batch, 2]
    x                      = torch.log_softmax(x, dim = 1) # [batch, 2]

    return x
```

In PyTorch, RNNs take inputs of shape `[sequence length, batch size,
embedding dimension]` as inputs for subtle computation efficiency
reasons.

After applying the model from left to right and right to left
(`bidirectional = True`), we get the output of the RNN in `hidden`. We
reshape it in order to feed it to a linear layer to obtain our final
predictions.

```
[   0] Train eval 50.259%, Test eval 47.667%
[   5] Train eval 67.444%, Test eval 60.000%
[  10] Train eval 77.111%, Test eval 64.333%
[  15] Train eval 83.519%, Test eval 66.333%
[  20] Train eval 89.556%, Test eval 69.333%
[  25] Train eval 93.815%, Test eval 69.333%
[  30] Train eval 93.852%, Test eval 67.333%
[  35] Train eval 98.778%, Test eval 71.667%
[  40] Train eval 99.333%, Test eval 72.000%
[  45] Train eval 94.556%, Test eval 65.333%
[  50] Train eval 99.704%, Test eval 71.333%
[  55] Train eval 99.889%, Test eval 70.667%
[  60] Train eval 100.000%, Test eval 70.667%
[  65] Train eval 100.000%, Test eval 70.333%
[  70] Train eval 100.000%, Test eval 70.667%
[  75] Train eval 100.000%, Test eval 71.000%
[  80] Train eval 100.000%, Test eval 70.000%
[  85] Train eval 100.000%, Test eval 70.000%
[  90] Train eval 99.815%, Test eval 73.667%
[  95] Train eval 100.000%, Test eval 72.667%
[ 100] Train eval 100.000%, Test eval 72.667%
[ 105] Train eval 100.000%, Test eval 73.000%
[ 110] Train eval 100.000%, Test eval 72.667%
[ 115] Train eval 100.000%, Test eval 72.000%
[ 120] Train eval 100.000%, Test eval 72.000%
[ 125] Train eval 100.000%, Test eval 72.333%
[ 130] Train eval 100.000%, Test eval 72.667%
[ 135] Train eval 100.000%, Test eval 72.000%
[ 140] Train eval 100.000%, Test eval 72.333%
[ 145] Train eval 100.000%, Test eval 72.333%
[ 150] Train eval 100.000%, Test eval 72.667%
```

We can see that these models are very powerful. They still overfit
strongly to the training data because of the dataset size but they
also achieve the best accuracies on the test set.
