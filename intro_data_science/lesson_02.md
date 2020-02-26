# Course 2

Pandas practical work, inspired by [Python Data Science
Handbook](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.00-Introduction-to-Pandas.ipynb)
and [10 minutes to
pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)

*Pandas objects* can be thought of as enhanced versions of NumPy
structured arrays in which the rows and columns are identified with
labels rather than simple integer indices.

As pandas capabilities are extremely vast we will only go over a few
of the main ones in this course. For more in depth explanations about
this library please take a look at the references.

## Pandas `Series`

A pandas `Series` is a one dimensional array with axis labels.

```python
>>> import pandas as pd # Standard way to import Pandas
>>> s = pd.Series([1, 3, 5, np.nan, 6, 8])
>>> s
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64

>>> s.values
array([ 1.,  3.,  5., nan,  6.,  8.])

>>> type(s.values)
numpy.ndarray

>>> s.index
RangeIndex(start=0, stop=6, step=1)

>>> s[1]
3.0

>>> s[1:3]
1    3.0
2    5.0
dtype: float64
```

We can see multiples things in this first example. The output contains
two columns, the first one corresponds to the `index` of the `Series`
and the second one corresponds to the `values` of the `Series`. We can
also see that the values are actually stored in a NumPy `ndarray`
which type has been set to `float64` because of the `NaN` presence
("_Not a number_" is a commonly used float value to indicate missing
values). We can also use standard 1D NumPy indexing to access the
values.

The `index` used in the previous example was simply a range of
integers. We can use other types of index to work with our data in a
more explicit manner.

```python
>>> s = pd.Series([1, 3, 5, np.nan, 6, 8], index = ['a', 'b', 'c', 'd', 'e', 'f'])
>>> s
Out[235]:
a    1.0
b    3.0
c    5.0
d    NaN
e    6.0
f    8.0
dtype: float64

>>> s['b']
3.0

>>> s['b': 'e'] # The elements from the index b to the index e
b    3.0
c    5.0
d    NaN
e    6.0
dtype: float64
```

## Pandas `DataFrame`

The Pandas `DataFrame` is probably the object that you will manipulate
the most while doing Data Science using Python. Like `Series`,
`DataFrame` can be thought of as a generalization of NumPy arrays.

There are multiple ways of creating a new `DataFrame`, the most useful
ones are presented in [this
page](https://pbpython.com/pandas-list-dict.html). Let's use
`from_records` in the next example. When using this method, we give it
a list of rows which are tuples of values.

```python
>>> states = pd.DataFrame.from_records([
    (38332521, 423967),
    (26448193, 695662),
    (19651127, 141297),
    (19552860, 170312),
    (12882135, 149995)
    ],
    index = [
        'California',
        'Texas',
        'New York',
        'Florida',
        'Illinois'
    ],
    columns = [
        'population',
        'area'
    ]
)
>>> states
            population    area
California    38332521  423967
Texas         26448193  695662
New York      19651127  141297
Florida       19552860  170312
Illinois      12882135  149995
```

Creating `DataFrame` by hand is actually quite rare when working with
pandas. Most often you will load a save file directly into a
`DataFrame`. Many [file
format](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)
are available, with the most common one probably being *CSV*
([Comma-separated
values](https://en.wikipedia.org/wiki/Comma-separated_values)). For
example if we want to load the file [save.csv](../datasets/save.csv):

```python
>>> df = pd.read_csv('save.csv')
>>> df
    a   b   c   d   e
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19
4  20  21  22  23  24
```

## Indexing data

Once our data is loaded in our `DataFrame`, we can access it using a
[variety of
methods](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing).

```python
>>> df
    a   b   c   d   e
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19
4  20  21  22  23  24

>>> df['a'] # Selecting the column 'a'
0     0
1     5
2    10
3    15
4    20
Name: a, dtype: int64

>>> df.a # Selecting the column 'a' using a notation shortcut
0     0
1     5
2    10
3    15
4    20
Name: a, dtype: int64

>>> df.loc[:, 'a': 'c'] # Selecting all the lines, with the columns from 'a' to 'c'
    a   b   c
0   0   1   2
1   5   6   7
2  10  11  12
3  15  16  17
4  20  21  22
```

There is multiple things to notice about the previous block of
code. The notation shortcut `df.a` to access `df['a']` is useful but
should be used carefully as column names may conflict with methods
name of the `DataFrame` class. The `loc` method is label based which
means that we can select elements based on the label of the
index. Another way to access your data is by using `iloc` which,
contrary to `loc`, is integer based. `iloc` is very similar to the
usual NumPy indexing behavior

```python
>>> df
    a   b   c   d   e
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19
4  20  21  22  23  24

>>> df.iloc[2: 4, 1 : 3] # Selecting rows 2 to 4 and columns from
                         # the index 1 to the index 3
    b   c
2  11  12
3  16  17
```

Just like with NumPy, we can select rows and columns using boolean
arrays. This is extremely useful and is used a lot when performing
data analysis.

```python
>>> df
    a   b   c   d   e
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19
4  20  21  22  23  24

>>> df[df.c >= 12] # Selecting the rows where the value in the column c is >= 12
    a   b   c   d   e
2  10  11  12  13  14
3  15  16  17  18  19
4  20  21  22  23  24

>>> df[(df.c >= 12) & (df.e % 2 == 0)] # Selecting the rows where the value
                                       # in the column c is >= 12 and the value in
                                       # the column e is even
    a   b   c   d   e
2  10  11  12  13  14
4  20  21  22  23  24
```

Exercises:
All the exercises use the following arrays:

```python
In [278]: df = pd.DataFrame(np.arange(25).reshape(5, 5), columns = ['a', 'b', 'c', 'd', 'e'])

In [279]: df
Out[279]:
    a   b   c   d   e
0   0   1   2   3   4
1   5   6   7   8   9
2  10  11  12  13  14
3  15  16  17  18  19
4  20  21  22  23  24
```

1. Select the rows where `d` is greater than or equal to `b + c`. The
   result should be

```python
   a  b  c  d  e
0  0  1  2  3  4
```

2. Compute the sum of the values in each column. The result should be

```python
a    50
b    55
c    60
d    65
e    70
dtype: int64
```

3. Select the columns `b` and `c` in two different ways. The result should be

```python
    b   c
0   1   2
1   6   7
2  11  12
3  16  17
4  21  22
```

2. (_harder_) Select the columns which sum is an even number. The result should be

```python
    a   c   e
0   0   2   4
1   5   7   9
2  10  12  14
3  15  17  19
4  20  22  24
```

## Handling missing data

A major part of data science is cleaning datasets in order to use them
as input to machine learning algorithms. One aspect of this cleaning
procedure is to ensure that the data does not contain missing values.

Let's take a look at the classical [Titanic
dataset](https://www.kaggle.com/c/titanic/overview) from Kaggle. This
dataset contains information about Titanic passengers (such as name,
gender, age, ...) and whether they died during the crash. This dataset
is often used to build simple machine learning models to predict the
survival of passenger based on their information. We will come back to
it when we will build machine learning models, for now let's clean it.

First let's download the file [here](../datasets/titanic.csv) and load it
using pandas.

```python
>>> import pandas as pd
>>> df = pd.read_csv('titanic.csv')
>>> df.shape
(891, 12)

>>> df.columns
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

>>> df.head()
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S

>>> df.describe()
       PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
```

The first things we usually do when discovering a new dataset is
getting to know it. The first thing we did is print its shape using
`df.shape`. The result is `(891, 12)` which tells us that we have 891
passengers and that we have 12 fields for each passenger. Then we take
a look at the kind of information we have access to by printing the
names of the columns using `df.columns`, a full description of each
column is available [here](https://www.kaggle.com/c/titanic/data). We
then take a look a the first 5 rows of the dataset using
`df.head()`. Finally, we use `df.describe` to print summary statistics
on the numerical fields of the dataset.

In this section, we focus on handling missing data. There are [many
ways](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)
to handle missing data using pandas, we will only take a look at few
in this section.

First, let's take a look at how many values are missing.

```python
In [10]: df.isnull()
Out[10]:
     PassengerId  Survived  Pclass   Name    Sex    Age  SibSp  Parch  Ticket   Fare  Cabin  Embarked
0          False     False   False  False  False  False  False  False   False  False   True     False
1          False     False   False  False  False  False  False  False   False  False  False     False
2          False     False   False  False  False  False  False  False   False  False   True     False
3          False     False   False  False  False  False  False  False   False  False  False     False
4          False     False   False  False  False  False  False  False   False  False   True     False
..           ...       ...     ...    ...    ...    ...    ...    ...     ...    ...    ...       ...
886        False     False   False  False  False  False  False  False   False  False   True     False
887        False     False   False  False  False  False  False  False   False  False  False     False
888        False     False   False  False  False   True  False  False   False  False   True     False
889        False     False   False  False  False  False  False  False   False  False  False     False
890        False     False   False  False  False  False  False  False   False  False   True     False

[891 rows x 12 columns]

In [11]: df.isnull().sum()
Out[11]:
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64

In [12]: df.isnull().sum() / len(df)
Out[12]:
PassengerId    0.000000
Survived       0.000000
Pclass         0.000000
Name           0.000000
Sex            0.000000
Age            0.198653
SibSp          0.000000
Parch          0.000000
Ticket         0.000000
Fare           0.000000
Cabin          0.771044
Embarked       0.002245
dtype: float64
```

`df.isnull` return a boolean DataFrame with `True` values where values
are missing. To know how many values are missing by column we use
`sum` and finally to get the proportion of missing value in each
column we broadcast a division by the number of rows in the dataset
`len(df)`. We can see that the 19.9% of the `Age` values, 77.1% of the
`Cabin` values and 0.225% of the `Embarked` values are missing.

Know that we know where values are missing we have to decide how to
handle it. There are multiple strategies to deal with this situation:

- We can remove rows containing missing values. This is quite extreme
  as in this case we would lose 77.1% of our dataset because of the
  `Cabin` column. We could also only apply this strategy to the
  `Embarked` column.
- We can remove columns containing too many missing values. This might
  be suitable for the `Cabin` column.
- We can add a new special value indicating the absence of information.
- We can fill the missing values using a valid value like the median
  for numerical fields or the value that appears the most for
  categorical fields.

It is important to note that there is no "right" or "wrong" way to
handle missing value, it all depends on the use case and what we want
to use our data for later.

Now that we know the available strategies, let's deal with our missing
values. We start with the `Embarked` column. Let's take a look at the
values this column takes.

```python
>>> df.Embarked.value_counts()
S    644
C    168
Q     77
Name: Embarked, dtype: int64
```

As the values are quite spread out across the different ports of
embarkation, we chose to drop the rows for which we do not have the
`Embarked` information. Once again this is not necessarily the best
strategy to handle it, we could also have filled the missing values
with `S`.

```python
>>> df = df.dropna(subset = ['Embarked'])

>>> df.isnull().sum() / len(df)
PassengerId    0.000000
Survived       0.000000
Pclass         0.000000
Name           0.000000
Sex            0.000000
Age            0.199100
SibSp          0.000000
Parch          0.000000
Ticket         0.000000
Fare           0.000000
Cabin          0.772778
Embarked       0.000000
dtype: float64
```

[`dropna`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html)
is a method that is used to drop rows or columns based on missing
values. In our case we would like to only drop the rows with the
`Embarked` information missing so we use the `subset` argument. If we
did not use it it would also have dropped all the rows with missing
`Cabin` number.

Now let's deal with the missing `Cabin` numbers. As 77.3% of the
column is missing we will just remove it.

```python
In [25]: df = df.drop(['Cabin'], axis = 1)

In [26]: df.isnull().sum() / len(df)
Out[26]:
PassengerId    0.0000
Survived       0.0000
Pclass         0.0000
Name           0.0000
Sex            0.0000
Age            0.1991
SibSp          0.0000
Parch          0.0000
Ticket         0.0000
Fare           0.0000
Embarked       0.0000
dtype: float64
```

We use the `drop` method to drop the column by precising that we drop
along the `axis` 1 (columns). We can also notice that by default most
`DataFrame` operation are not done in place but a new `DataFrame` is
created, we can alter this behavior by giving `inplace = True` to
modification methods.

Now let's deal with the Age column. We will fill the missing values
with the median age. We often use the median instead of the mean as it
is more robust to outliers (extremely small or big values) of the
dataset.

```python
>>> df.Age.fillna(df.Age.median(), inplace = True)
>>> df.isnull().sum() / len(df)
PassengerId    0.0
Survived       0.0
Pclass         0.0
Name           0.0
Sex            0.0
Age            0.0
SibSp          0.0
Parch          0.0
Ticket         0.0
Fare           0.0
Embarked       0.0
dtype: float64
```

We use `fillna` to fill the missing value with a specific one, in this
case `df.Age.median()`. Here we use `inplace = True` as, contrary to
`dropna`, `fillna` does not have a `subset` argument.

## Combining datasets

Similarly to NumPy, we can concatenate multiple `DataFrame`s:

```python
>>> df_rand = pd.DataFrame(np.random.randn(10, 4))
>>> df_rand
          0         1         2         3
0 -0.591078 -0.041216  0.245512 -1.241936
1  1.488797 -0.508046  0.666062  1.518212
2  0.227398 -1.641177  0.038620 -1.110990
3  0.205498 -0.505979  1.543662 -0.523256
4 -1.100840  0.606044  0.645516 -1.111447
5 -0.031574  1.539560 -0.283304  0.970026
6 -0.633329 -0.536459  0.874104  0.439670
7 -0.604515 -0.778099 -0.057865  0.189544
8 -0.030452 -0.954243 -0.097178 -0.283184
9 -0.075592 -1.962462  0.916878  0.689099

>>> pieces = [df_rand[:3], df_rand[3:7], df_rand[7:]]
>>> pd.concat(pieces)
          0         1         2         3
0 -0.591078 -0.041216  0.245512 -1.241936
1  1.488797 -0.508046  0.666062  1.518212
2  0.227398 -1.641177  0.038620 -1.110990
3  0.205498 -0.505979  1.543662 -0.523256
4 -1.100840  0.606044  0.645516 -1.111447
5 -0.031574  1.539560 -0.283304  0.970026
6 -0.633329 -0.536459  0.874104  0.439670
7 -0.604515 -0.778099 -0.057865  0.189544
8 -0.030452 -0.954243 -0.097178 -0.283184
9 -0.075592 -1.962462  0.916878  0.689099
```

We can add a new column to a dataset by performing an affectation

```python
>>> df_rand[5] = df_rand[0] + df_rand[2]
>>> df_rand
          0         1         2         3         5
0 -0.591078 -0.041216  0.245512 -1.241936 -0.345567
1  1.488797 -0.508046  0.666062  1.518212  2.154858
2  0.227398 -1.641177  0.038620 -1.110990  0.266018
3  0.205498 -0.505979  1.543662 -0.523256  1.749160
4 -1.100840  0.606044  0.645516 -1.111447 -0.455324
5 -0.031574  1.539560 -0.283304  0.970026 -0.314878
6 -0.633329 -0.536459  0.874104  0.439670  0.240775
7 -0.604515 -0.778099 -0.057865  0.189544 -0.662380
8 -0.030452 -0.954243 -0.097178 -0.283184 -0.127630
9 -0.075592 -1.962462  0.916878  0.689099  0.841287
```

We can also append new records to an existing `DataFrame`.

```python
>>> states = pd.DataFrame.from_records([
    (38332521, 423967),
    (26448193, 695662),
    (19651127, 141297),
    (19552860, 170312),
    (12882135, 149995)
    ],
    index = [
        'California',
        'Texas',
        'New York',
        'Florida',
        'Illinois'
    ],
    columns = [
        'population',
        'area'
    ]
)
>>> states
            population    area
California    38332521  423967
Texas         26448193  695662
New York      19651127  141297
Florida       19552860  170312
Illinois      12882135  149995

>>> new_hamphire = pd.Series(
    [1356458, 24214],
    index = ['population', 'area'],
    name = 'New Hampshire'
)
>>> states.append(new_hamphire)
               population    area
California       38332521  423967
Texas            26448193  695662
New York         19651127  141297
Florida          19552860  170312
Illinois         12882135  149995
New Hampshire     1356458   24214
```

And finally, we can combine two datasets together by performing a SQL
style merge. This kind of operation is very powerful and supports a
lot of different behaviors, for more information refer to the
[official
documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#merging-join).

```python
>>> left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
>>> right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
>>> left
   key  lval
0  foo     1
1  foo     2

>>> right
   key  rval
0  foo     4
1  foo     5

>>> pd.merge(left, right, on = 'key')
   key  lval  rval
0  foo     1     4
1  foo     1     5
2  foo     2     4
3  foo     2     5
```

## Groupby

By *group by* we are referring to a process involving one or more of
the following steps:

- Splitting the data into groups based on some criteria
- Applying a function to each group independently
- Combining the results into a data structure

The group by in pandas is used extremely often to perform various kind
of computations and explorations. Let's get back to the Titanic
dataset that we cleaned earlier and explore it using groupbys.

We can for example compute the average age of men and women:

```python
>>> df.groupby('Sex').Age.mean()
Sex
female    27.788462
male      30.140676
Name: Age, dtype: float64
```

With `df.groupby('Sex')` we choose to decompose our dataset according
to the `Sex` column. With `.Age` we select the `Age` column in each of
the groups and then we compute the `mean`. The result will be indexed
by all the possible values for the `groupby` key.

Now let's illustrate the groupby potential by performing a bit of
exploration. A question that we can ask is "Have the [Women and
children
first](https://en.wikipedia.org/wiki/Women_and_children_first) code of
conduct been applied during the Titanic crash?". Let's see.

When performing a group by operation, we are not limited to a single
key, we can separate groups according to multiple columns.

```python
>>> df.groupby(['Sex', 'Survived']).size()
Sex     Survived
female  0            81
        1           231
male    0           468
        1           109
dtype: int64
```

Here, we ask how many men and women have survived and died. The result
is a `Series` indexed by a [hierarchical
index](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html):
Each line corresponds to a couple `(Sex, Survived)`. A better way to
present this result would be to display the proportion of men and
women that died, we can use another groupby to perform this computation.

```python
>>> df.groupby('Sex').size()
Sex
female    312
male      577
dtype: int64

>>> df.groupby(['Sex', 'Survived']).size() / df.groupby('Sex').size()
Sex     Survived
female  0           0.259615
        1           0.740385
male    0           0.811092
        1           0.188908
dtype: float64
```

Here we can see that 74% of the women on the Titanic did survive
while only 18.9% of the men did.

Now let's take a look at the age repartition by
[quantiles](https://en.wikipedia.org/wiki/Quantile).

```python
>>> np.arange(0, 1.1, .1)
array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

>>> df.Age.quantile(np.arange(0, 1.1, .1))
0.0     0.42
0.1    16.00
0.2    20.00
0.3    24.00
0.4    28.00
0.5    28.00
0.6    28.00
0.7    32.00
0.8    38.00
0.9    47.00
1.0    80.00
Name: Age, dtype: float64
```

We see that 10% of the people are 16 years old or younger. We will
consider all the passengers in this category as children. Let's add a
column to the dataset indicating whether a passenger is a child or
not.

```python
>>> df['IsChild'] = df.Age <= df.Age.quantile(0.1)
>>> df.IsChild.sum() / len(df)
0.1124859392575928
```

Now that we know which passengers are children, let's look at their
survival rate.

```python
>>> df.groupby(['IsChild', 'Survived']).size()
IsChild  Survived
False    0           504
         1           285
True     0            45
         1            55
dtype: int64

>>> df.groupby(['IsChild', 'Survived']).size() / df.groupby('IsChild').size()
IsChild  Survived
False    0           0.638783
         1           0.361217
True     0           0.450000
         1           0.550000
dtype: float64
```

We see that 55% of children have survived as opposed to 36.1% of
adults.

Exercises:
1. Compute the mean `Fare` by passenger class (`Pclass`).

2. Compute, by embarkation port the number of passenger in each passenger class.

3. Compute the survival rate by passenger class.

4. In order to test the "Women and children first" code of conduct,
  compute the survival rate of passengers that are children or female.

5. Compute the survival rate of people that are in the 20% oldest, are
   `male` and are not in the first passenger class.
