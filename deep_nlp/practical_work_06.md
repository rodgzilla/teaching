# Practical work 6

## Apply BERT to horoscope classification

Apply the BERT model to [this
dataset](../datasets/horoscope_dataset.csv). The task consists in
determining the sign to which an horoscope is about. The horoscope are
evenly spread among 12 signs giving a baseline accuracy of 8.3%. Using
BERT you should get an accuracy of around 35 to 40%.

```
SIGN       DATE        TEXT
cancer     2013-04-03  You gain strength as the day progresses now th...
pisces     2011-07-19  The Moon's monthly visit to your sign usually ...
gemini     2011-03-08  You might have a lot on your mind today, makin...
libra      2014-10-09  An unresolved emotional issue may stubbornly w...
virgo      2012-03-02  You want to be more involved with everyone at ...
cancer     2013-01-21  Sharing your emotions can be a challenge now, ...
aries      2014-09-14  Your intuition tells you to follow your feelin...
taurus     2010-08-21  You are very aware of the difference between h...
virgo      2017-05-06  You don't like being pushed to the edge becaus...
capricorn  2014-08-23  You can see a relationship meltdown in the mak...
```

## Language model

Using the [dataset](../datasets/horoscope_dataset.csv) presented
above, build an automatic horoscope generator using the language
modeling task.

Let's consider the following horoscope dataset:

```
["Although you may have looked forward to doing something special today, it may not sound so good as the time approaches. There's a part of you that just feels lazy and would rather be a couch potato. But even if it takes a while to get going, you'll feel better about everything once you are in motion. Give yourself enough time to rest, but don't let the day pass you by without burning off a bit of physical energy.",
 "You are eager to kick back and indulge yourself today, but it may turn into a much more active day than you expected. A flash of inspiration could set you on a path toward a simple goal. But once you get started your project begins to grow. Be careful, for it's easy to think you are being practical when you are actually making your life more complicated than necessary.",
 "You may want to get a jump-start on financial planning for the year ahead, especially if you just need some time to reevaluate your goals. But instead of dreaming about how you can make more money, it's smarter to consider ways to limit excess spending. Creating a budget based upon known expenses can help you figure out what your options actually are now. Even if you are only guessing at this point, it's still a great exercise to get you thinking in a more responsible manner.",
 "If you want some peace and quiet today, you might have to come right out and tell someone to leave you alone. But how you deliver the message can be just as important as what you say. Make it clear that you are not rejecting others, because you may not get your much-needed silence if they take your request personally. You'll have more to give those you love once you get some rest.",
 "This can be a great day to do something special for yourself that establishes a pattern for a new daily routine. Keep in mind that your gift doesn't have to be extravagant; it only needs to be healthy. Consider a small step that's related to exercise or diet that you can take now. You might be pleasantly surprised at how a little change can make such a big difference.",
 "If you are feeling restricted by someone in your life today, an unexpected turn of events can release a bit of tension and loosen the reins. Paradoxically, this surprising twist might be provoked by a misunderstanding that forces you to talk about what's happening. Even if you are uncomfortable with where the conversation is going at first, stick with it. Luckily, unforeseen relief is on the way.",
 "Normally, you're a better team player than a solo act, but today you may be driven to show everyone how capable you are of managing things. Others might not realize what you're doing now, for you can be quite subtle as you exercise control over what's happening. Keep in mind that you can make it much easier on yourself once you realize that you don't have to take charge of everything.",
 "You can make it appear as if you don't want to participate in the activities of the day, yet you might still be hurt if you are not included. You cannot blame others if they believe they are doing what you prefer by leaving you alone. Fortunately, you can control the outcome by being very clear about what you want.",
 "It's difficult for you to establish healthy limits today as the Sun activates your key planet Jupiter. You mistakenly believe that if a small quantity is good then a larger amount is even better. But this kind of logic will land you in a heap of trouble if you have to deal with the consequences of overindulgence later in the day. Be smart and stop before it's too late.",
 "You have some serious magic working in your favor today, but you must figure out how to harness it. If you cannot find something constructive to do with all your energy, your frustration could build until you lose your temper at someone who probably isn't even the real source of your annoyance. Remember that others are relaxing on this holiday, so don't try to impose your ambitious plans on them just yet."]
```

The steps are the following ones:

- Concatenate all the horoscope together to form a large text.

```
"Although you may have looked forward to doing something special today, it may not sound so good as the time approaches. There's a part of you that just feels lazy and would rather be a couch potato. But even if it takes a while to get going, you'll feel better about everything once you are in motion. Give yourself enough time to rest, but don't let the day pass you by without burning off a bit of physical energy. You are eager to kick back and indulge yourself today, but it may turn into a much more active day than you expected. A flash of inspiration could set you on a path toward a simple goal. But once you get started your project begins to grow. Be careful, for it's easy to think you are being practical when you are actually making your life more complicated than necessary. You may want to get a jump-start on financial planning for the year ahead, especially if you just need some time to reevaluate your goals. But instead of dreaming about how you can make more money, it's smarter to consider ways to limit excess spending. Creating a budget based upon known expenses can help you figure out what your options actually are now. Even if you are only guessing at this point, it's still a great exercise to get you thinking in a more responsible manner. If you want some peace and quiet today, you might have to come right out and tell someone to leave you alone. But how you deliver the message can be just as important as what you say. Make it clear that you are not rejecting others, because you may not get your much-needed silence if they take your request personally. You'll have more to give those you love once you get some rest. This can be a great day to do something special for yourself that establishes a pattern for a new daily routine. Keep in mind that your gift doesn't have to be extravagant; it only needs to be healthy. Consider a small step that's related to exercise or diet that you can take now. You might be pleasantly surprised at how a little change can make such a big difference. If you are feeling restricted by someone in your life today, an unexpected turn of events can release a bit of tension and loosen the reins. Paradoxically, this surprising twist might be provoked by a misunderstanding that forces you to talk about what's happening. Even if you are uncomfortable with where the conversation is going at first, stick with it. Luckily, unforeseen relief is on the way. Normally, you're a better team player than a solo act, but today you may be driven to show everyone how capable you are of managing things. Others might not realize what you're doing now, for you can be quite subtle as you exercise control over what's happening. Keep in mind that you can make it much easier on yourself once you realize that you don't have to take charge of everything. You can make it appear as if you don't want to participate in the activities of the day, yet you might still be hurt if you are not included. You cannot blame others if they believe they are doing what you prefer by leaving you alone. Fortunately, you can control the outcome by being very clear about what you want. It's difficult for you to establish healthy limits today as the Sun activates your key planet Jupiter. You mistakenly believe that if a small quantity is good then a larger amount is even better. But this kind of logic will land you in a heap of trouble if you have to deal with the consequences of overindulgence later in the day. Be smart and stop before it's too late. You have some serious magic working in your favor today, but you must figure out how to harness it. If you cannot find something constructive to do with all your energy, your frustration could build until you lose your temper at someone who probably isn't even the real source of your annoyance. Remember that others are relaxing on this holiday, so don't try to impose your ambitious plans on them just yet."
```

- Choose a context size slide a window of that size on your text to
  generate your dataset.

```python
>>> window_size = 10
...
>>> inputs, outputs = split_text(text, window_size)
>>> inputs[0]
['Although',
 'you',
 'may',
 'have',
 'looked',
 'forward',
 'to',
 'doing',
 'something',
 'special']

>>> outputs[0]
'today,'

>>> inputs[-1]
["don't",
 'try',
 'to',
 'impose',
 'your',
 'ambitious',
 'plans',
 'on',
 'them',
 'just']

>>> outputs[-1]
'yet.'
```

- Build a vocabulary, a token to id and an id to token correspondence as
  we did during the previous course.

```python
>>> list(vocabulary)[:10]
['forces', "it's", 'You', 'Luckily,', 'careful,', 'but', 'great', 'over', 'with', 'hurt']

>>> token_to_idx['subtle']
332

>>> token_to_idx['life']
302

>>> idx_to_token[302]
'life'

>>> idx_to_token[332]
'subtle'
```

- Encode the inputs and the outputs.

```python
>>> encoded_inputs[0]
[142, 15, 289, 223, 155, 143, 147, 33, 40, 333]

>>> ' '.join(idx_to_token[idx] for idx in encoded_inputs[0])
'Although you may have looked forward to doing something special'

>>> encoded_outputs[0]
237

>>> idx_to_token[237]
'today,'

>>> outputs[0]
'today,'
```

- Wrap your encoded inputs and outputs into a `TensorDataset`.

- Split your dataset into a training and evaluation set.

- Build a convolutional text classifier with the correct number of
  output classes that you will use to model the language.

- Train and evaluate your model.

- Build a function that generates a fake horoscope word by word
  starting with a prefix given as argument. The final length of the
  sequence is given as argument to the function.
