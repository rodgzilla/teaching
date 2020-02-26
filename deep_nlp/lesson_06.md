# Lesson 6

## NLP Pre-training tasks

### Language modeling

As mentioned earlier in the course, a small revolution has been
happening in the machine learning world with the discovery of very
powerful transfer learning method for Natural Language Processing
(NLP) tasks. These advances are mainly explained in the following
research articles

- [Universal Language Model Fine-tuning for Text
  Classification](https://arxiv.org/abs/1801.06146),
- [Improving Language Understanding by Generative
  Pre-Training](https://openai.com/blog/language-unsupervised/)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).

All advances have in common the kind of task used to pretrained
model. When we performed transfer learning in computer vision, we used
an ImageNet network such as *VGG16* or *ResNet* as a starting
point. This worked pretty well because ImageNet classification is a
task for which we have a huge amount of training data and involves a
wide variety of real word patterns. The task used in the research
article mentioned above has the same property.

![Language modeling](../figures/language_modeling.png)

The *language modeling* task consists in taking a sequence of words as
input and to predict the most likely following word. It can be seen as
a *classification task*, the number of classes being the number of words
in the vocabulary.

The figure presented above shows a basic example of language modeling
in which the model has to predict a single words using only words that
appeared in the "past" (to its left). More advanced techniques such as
the *masked language modeling* task used to pretrain the BERT model
make the model predict a variable number of words, using context from
the left and the right (bidirectional property).

This task is very choice for multiple reasons:

- We have nearly unlimited training data for it, (inputs, output)
  couples can be extracted for every texts in the correct language.
- Performing this task correctly requires to have at least some
  semantic understanding of all the concepts mentioned in the
  context. In the figure above, we can see that understanding the
  meaning and the context in which the term `but` is used is necessary
  to produce the right prediction.

### Next sentence prediction

In the [BERT](https://arxiv.org/abs/1810.04805) article, the authors
also use the *next sentence prediction* task. This task consists in
taking a pair sentences and predicting whether they appear next to
other in a text. Once again, we have huge amount of training data for
this task.

The language modeling task is great to force the model to capture
*word-level semantics* but it does not focus a lot on relationship
between two sentences. Capturing this kind of *sentence-level
semantics* is very useful for many downstream tasks such as Question
Answering (QA) and Natural Language Inference (NLI).

## BERT

### Theory

#### Attention mechanism

The model architecture used in the BERT article is called a
*Transformer Network*. This family of models have been introduced in
the article [Attention Is All You
Need](https://arxiv.org/abs/1706.03762) so, in order to study how this
model works, we first have to look at *Attention Mechanisms*.

Attention mechanisms are inspired by the idea that, when humans take
decisions based on visual information, they focus most of their
*attention* on a specific portion of the available information. Let's
take a look at a result of such mechanism being implemented in a
neural network. This example is from the article [Show, Attend and
Tell: Neural Image Caption Generation with Visual
Attention](https://arxiv.org/abs/1502.03044) in which the authors
create a neural network which generates captions for images using
an attention mechanism.

![Visual attention](../figures/visual_attention_concept.png)

This figures shows for the underlined word the parts of the image on
which the neural networks *focused* its attention to choose this
word. By allowing a model to choose the weight it will give to each
part of its inputs in its computations, we greatly improve its
computing power.

The same concept applies to natural language processing tasks.

![Translation attention](../figures/translation_attention.jpeg)

(image from [Floydhub](https://blog.floydhub.com/attention-mechanism/))

Let's go over an example of *Scaled Dot-Product Attention*

![Detailed translation attention](../figures/detailed_attention.png)

In this example, we are translating a sentence from french to
English. The french sentence is "Elle alla à la plage" and so far our
model have generated the following output sentence "She went to
the". The model now has to choose what the next word should be.

The model has an internal representation of each of the token in the
input and in the output, we can think of this representation as an
*embedding* of the token that has been computed by the model. Now
using this embedding, the *attention layer* will use three linear
layers to produced three tensors, a *query* tensor, a *key* tensor and
a *value* tensor.

The *query* tensor is computed using the translation. It represents
what *type* of information we are looking for to generate the next
word. Based on the beginning of the sentence "She went to the", the
*query layer* have computed that we want a *location* information.

The *key* tensor is computed using the input sentence. It represents the
*type* of information each of the input token represents. The *key
layer* have computed that the token "plage" contains a location
information.

The *value* tensor is computed using the input sentence. It represents
the *semantic information* contained in each token. For example, the
*value layer* have computed that the token "alla" refers to the
concept of the verb "to go" and is conjugated to the past tense and
that "plage" refers to the concept of "beach".

Now, to compute how much the model should focus on each part of the
input, `Attention(Q, K, V)`. To find what kind is the information that
we need, we compute the dot-product between the query (what we look
for) and the key tensor (what kind of information is available). Then,
to focus the attention mainly on a single place, we apply a softmax
activation. Now that we know *where* is the information we want, we
fetch its value by multiplying the result of the previous computation
to the value tensor. The scaling factor in the softmax layer is very
important theorically, you can find an explanation for its presence in
the section 3.2.1 of the "Attention is all you need" article.

You can take a look at
[this](https://distill.pub/2016/augmented-rnns/) distil.pub
publication to get more details on how the attention mechanism
functions on RNNs or
[this](http://akosiorek.github.io/ml/2017/10/14/visual-attention.html)
blog post from Adam Kosiorek.

#### BERT embeddings

The BERT model uses three different kinds of embeddings.

![BERT Embeddings](../figures/bert_embeddings.png)

The *token embeddings* corresponds to classical word embeddings as we
have seen in the previous course. These are 768 dimensional vectors
associated to each token of the vocabulary encoding its *semantic
meaning*.

The BERT architecture is applicable to many tasks, some of them
taking more than one sequence as input. Some examples include the
"Next sentence prediction" task and the Textual semantic similarly task
(quantifying how much two sentences are semantically similar). In
order to allow the model to treat the first and the second sequence
differently, we add *segment embeddings* indicating to what sequence
belongs the current token.

Because all the internal layers of the BERT model are *linear* and
uses attention layer, the model looses by default information about
the *order* of the tokens in the input. As the token order information
is very important, we add *position embeddings*. Each position of the
input has a specific embedding that have been trained during the
pre-training phase. The model has learned to encode the position of
the words in the sequences in order to use it to improve its
performances.

Let's take an example in the figure above. We have two sentences `my
dog is cute` and `he likes playing` and we want to predict if the
second comes after the first one in a text. When we encode the token
`likes`, we fetch three embeddings:
- The *token embedding* _E<sub>likes</sub>_ of `likes`, containing
  semantic information about this token
- `likes` comes from the second sentences so we fetch the segment
  embedding _E<sub>B</sub>_ of the second sentence
- `likes` is located at the position `7` in the input so we fetch the
  position encoding of the position `7`, _E<sub>7</sub>_

Once we have these three embeddings, we simply add them and pass the
result as input to the network.

#### Multi-task model

As explained before, BERT has been created to allow it to perform many
different kinds of tasks.

![Multi-task BERT](../figures/multi_task_bert.png)

Depending on the task we want to perform, we will feed the data and
read the predictions differently from the model.

In this course, we will only see the simplest case, the single
sentence classification tasks. To use BERT to perform this kind of
task, we first add the `[CLS]` token at the beginning of our
sequence. We then compute the embedding of our input as explained in
the previous section. As we only have a single sentence, all the
segment embeddings will be identical. BERT outputs a tensor for each
of the tokens of our input. To get a single representation of our
sequence, we select the tensor corresponding the the `[CLS]` token
(the first one) and feed it to a linear layer performing our
classification task.

### Model architecture

The BERT architecture is a bidirectional Transformer encoder from the
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).

![Transformer encoder](../figures/transformer_encoder.png)

Let's examine the output we get when we print the architecture of our
BERT model.

```python
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )

        [...]

        (11): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
```

### Code example

Let's finetune a BERT model to the sentiment classification that we
worked on during the previous course.

First we start by installing the `transformers` module.

```
pip install transformers
```

We then import everything we will need for our task.

```python
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.utils import InputExample, InputFeatures
```

We are going to create all the functions we will need to apply to our
task.

First we need to load a pre-trained version of the BERT model. There
exist a [large
variety](https://huggingface.co/transformers/pretrained_models.html)
of pre-trained BERT models, here are some of them:

- `bert-base-uncased`: 12-layer, 768-hidden, 12-heads, 110M
  parameters. Trained on lower-cased English text.
- `bert-large-uncased`: 24-layer, 1024-hidden, 16-heads, 340M
  parameters.  Trained on lower-cased English text.
- `bert-base-chinese`: 12-layer, 768-hidden, 12-heads, 110M parameters.
  Trained on cased Chinese Simplified and Traditional text.
- `bert-base-multilingual-cased`: 12-layer, 768-hidden, 12-heads, 110M
  parameters.  Trained on cased text in the top 104 languages with the
  largest Wikipedias.

For our task, we will use `bert-base-uncased`.

```python
def load_pretrained_model(pretrained_model_name, num_labels = 2):
  config = BertConfig.from_pretrained(
    pretrained_model_name,
    do_lower_case = True,
    num_labels    = num_labels
  )
  tokenizer = BertTokenizer.from_pretrained(
      pretrained_model_name,
      do_lower_case = True
  )
  model = BertForSequenceClassification.from_pretrained(
      pretrained_model_name,
      from_tf = False,
      config  = config
  )

  return config, tokenizer, model
```

We now write a function to load our dataset into the `transformers`
preferred format. No preprocessing is applied at this step, it will
come when we transform our *raw inputs* into *features*.

```python
def load_examples(filename):
  df            = pd.read_csv(filename, sep = '\t', names = ['comment', 'label'])
  df['comment'] = df.comment.str.lower()

  examples = [
      InputExample(
          guid   = sample_id,
          text_a = row.comment,
          label  = row.label
      )

      for sample_id, row in df.iterrows()
  ]

  return examples
```

Transformer network can take variable size inputs but just as with the
CNNs, every sequence in a batch must have the same size in order to be
able to wrap it with `torch.tensor`s. We will pad (add `[PAD]` tokens)
to every sequence of the dataset to fix this problem.

The feature generation step when using this library is very similar to
what we did during the previous course:

Let's recall the preprocessing steps we applied in the last course:

- First we split our sequences into list of tokens during the
  *tokenization* phase
- We truncate the sequences whose length are bigger than the maximum
  length that we have fixed.
- We padded the sequences to make them all have the same length.
- We created a *vocabulary*, the set of tokens that can appear in our
  inputs
- We *encoded* our sequences by replacing each token by its index in
  the vocabulary

These steps are also required when working with BERT, except that some
of them are done by the library and others have been done during the
pre-trained of the model.

```python
def convert_examples_to_features(examples, tokenizer, max_length):
  features = []
  for ex_index, example in enumerate(examples):
    inputs = tokenizer.encode_plus(
        text               = example.text_a,
        add_special_tokens = True,
        max_length         = max_length
    )
    input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
    # The attention mask is used to forbid BERT to focus its attention
    # on the padding tokens representations.
    attention_mask            = [1] * len(input_ids)

    # Padding everything
    pad_token            = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 0
    padding_length       = max_length - len(input_ids)
    input_ids            = input_ids      + [pad_token]            * padding_length
    token_type_ids       = token_type_ids + [pad_token_segment_id] * padding_length
    attention_mask       = attention_mask + [0]                    * padding_length

    features.append(
        InputFeatures(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            label          = example.label
        )
    )

  return features
```

The tokenization, truncation and encoding of the sequences are all
done by the call to `tokenizer.encode_plus`. The vocabulary of tokens
have already been created during the pre-training phase of the model.

As BERT takes 3 sequences as input (token ids, attention mask and
segment id), we have to pad all of them. An `InputFeatures` instance
simply is a class that groups together the tokens ids, attention mask,
segment ids and label of a sample.

Now that we have an `InputFeatures` instance for each sample of the
dataset, let's create a `TensorDataset` to wrap all the information.

```python
def create_dataset(features):
  all_input_ids      = torch.tensor([f.input_ids for f in features], dtype = torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype = torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype = torch.long)
  all_labels         = torch.tensor([f.label for f in features], dtype = torch.long)

  dataset = TensorDataset(
      all_input_ids,
      all_attention_mask,
      all_token_type_ids,
      all_labels
  )

  return dataset
```

As usual, we will split our dataset into a training set and and
evaluation set to properly assess the quality of our model
predictions.

```python
def split_dataset(dataset, test_prop):
  test_size = int(len(dataset) * test_prop)
  (
      train_dataset,
      test_dataset
  )         = random_split(dataset, [len(dataset) - test_size, test_size])

  return train_dataset, test_dataset
```

The evaluation function should look very similar to the one of every
classification problems we have seen until now. The only difference is
that the loss computation is taken care of by the model itself. It
knows what type of loss should be applied (`nn.NLLLoss`) because we
have loaded a `transformers.BertForSequenceClassification` pretrained
instance.

```python
def evaluate(model, device, loader, n_batch = 20):
  losses       = []
  correct_pred = 0
  total_pred   = 0
  model.eval()
  with torch.no_grad():
      for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        if step == n_batch:
          break
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels         = labels.to(device)

        loss, preds = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels         = labels
        )
        correct_pred += (preds.argmax(axis = 1) == labels).sum().item()
        total_pred   += len(labels)
        losses.append(loss.item())

  mean_loss = sum(losses) / len(losses)
  accuracy  = correct_pred / total_pred

  return mean_loss, accuracy
```

The training function also looks very similar to what we have seen in
the past. Once again, the model computes its loss by itself.

```python
def train(model, device, epochs, optimizer, scheduler, train_loader, test_loader):
  for epoch in range(epochs):
    for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
      model.train()
      input_ids      = input_ids.to(device)
      attention_mask = attention_mask.to(device)
      token_type_ids = token_type_ids.to(device)
      labels         = labels.to(device)

      outputs = model(
          input_ids      = input_ids,
          attention_mask = attention_mask,
          token_type_ids = token_type_ids,
          labels         = labels
      )
      loss = outputs[0]

      loss.backward()
      optimizer.step()
      scheduler.step()
      model.zero_grad()

      if step % 50 == 0:
        train_mean_loss, train_accuracy = evaluate(model, device, train_loader)
        eval_mean_loss, eval_accuracy   = evaluate(model, device, test_loader)
        print(
            f'[{epoch:2d}, {step:5d}]'
            f' Train: loss {train_mean_loss:6.3f}, '
            f'accuracy {100 * train_accuracy:5.2f}%    '
            f'|| Eval: loss {eval_mean_loss:6.3f}, '
            f'accuracy {100 * eval_accuracy:6.2f}%'
        )
```

The main function calls all the functions that we defined earlier in
order to run the model finetuning procedure.

```python
def main():
  # Hyperparameters setting
  pretrained_model_name = 'bert-base-uncased'
  device                = torch.device('cuda')
  batch_size            = 8
  lr                    = 2e-5
  adam_eps              = 1e-8
  epochs                = 3
  warmup_steps          = 0
  test_prop             = .1

  # Pretrained model loading
  (
    config,
    tokenizer,
    model
  )     = load_pretrained_model(pretrained_model_name)
  model = model.to(device)

  # Dataset loading
  examples         = load_examples('comments.txt')
  features         = convert_examples_to_features(examples, tokenizer, 75)
  dataset          = create_dataset(features)
  (
      train_dataset,
      test_dataset
  )                = split_dataset(dataset, test_prop)
  train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  test_loader  = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

  # Setup optimizer and learning rate scheduler
  optimizer            = AdamW(
      params = model.parameters(),
      lr     = lr,
      eps    = adam_eps
  )
  total_training_steps = len(train_loader) * epochs
  scheduler            = get_linear_schedule_with_warmup(
      optimizer          = optimizer,
      num_warmup_steps   = warmup_steps,
      num_training_steps = total_training_steps
  )

  train(model, device, epochs, optimizer, scheduler, train_loader, test_loader)

  return device, model, tokenizer, train_loader, test_loader
```

```
[ 0,     0] Train: loss  0.702, accuracy 53.75%    || Eval: loss  0.688, accuracy  55.62%
[ 0,    50] Train: loss  0.311, accuracy 92.50%    || Eval: loss  0.331, accuracy  93.12%
[ 0,   100] Train: loss  0.215, accuracy 93.75%    || Eval: loss  0.218, accuracy  93.12%
[ 0,   150] Train: loss  0.186, accuracy 92.50%    || Eval: loss  0.188, accuracy  93.75%
[ 0,   200] Train: loss  0.136, accuracy 95.00%    || Eval: loss  0.211, accuracy  91.88%
[ 0,   250] Train: loss  0.081, accuracy 98.12%    || Eval: loss  0.208, accuracy  92.50%
[ 0,   300] Train: loss  0.037, accuracy 99.38%    || Eval: loss  0.126, accuracy  95.00%
[ 1,     0] Train: loss  0.059, accuracy 98.75%    || Eval: loss  0.140, accuracy  96.25%
[ 1,    50] Train: loss  0.055, accuracy 98.12%    || Eval: loss  0.203, accuracy  95.00%
[ 1,   100] Train: loss  0.086, accuracy 97.50%    || Eval: loss  0.198, accuracy  94.38%
[ 1,   150] Train: loss  0.047, accuracy 98.12%    || Eval: loss  0.164, accuracy  95.62%
[ 1,   200] Train: loss  0.067, accuracy 98.12%    || Eval: loss  0.175, accuracy  93.75%
[ 1,   250] Train: loss  0.092, accuracy 98.12%    || Eval: loss  0.136, accuracy  94.38%
[ 1,   300] Train: loss  0.048, accuracy 98.75%    || Eval: loss  0.155, accuracy  95.00%
[ 2,     0] Train: loss  0.038, accuracy 99.38%    || Eval: loss  0.150, accuracy  92.50%
[ 2,    50] Train: loss  0.028, accuracy 99.38%    || Eval: loss  0.226, accuracy  93.12%
[ 2,   100] Train: loss  0.009, accuracy 100.00%   || Eval: loss  0.229, accuracy  92.50%
[ 2,   150] Train: loss  0.040, accuracy 99.38%    || Eval: loss  0.223, accuracy  93.12%
[ 2,   200] Train: loss  0.008, accuracy 100.00%   || Eval: loss  0.164, accuracy  94.38%
[ 2,   250] Train: loss  0.023, accuracy 98.75%    || Eval: loss  0.166, accuracy  93.12%
[ 2,   300] Train: loss  0.005, accuracy 100.00%   || Eval: loss  0.124, accuracy  95.00%
```

When finetuning BERT models, the learning rate and number of epochs
are both very low (`3` epochs and a learning rate of `2e-5` in our
case) as advised in the research article. Running more training epochs
usually does not improve the performances beyond what is reached
during the first few.
