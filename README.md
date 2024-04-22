# Icelandic homograph classification (IceHoc)

This project implements the classification of Icelandic homographs in a manner similar to that described in:

`Nicolis, M., Klimkov, V. (2021) Homograph disambiguation with contextual word embeddings for TTS systems. Proc. 11th
ISCA Speech Synthesis Workshop (SSW 11), 222-226, doi: 10.21437/SSW.2021-39`, utilizing contextual word embeddings.

However, it introduces some distinct modifications:

We employ logistic regression on embeddings produced by a transformer model to classify homographs.
We focus only on a specific "area" of tokens surrounding the homograph itself. This is configurable for training via the
`--around` parameter. We use the **CLS** token embedding as feature input and additionally, we incorporate the homograph
word embedding itself as input features for the classifier. To accommodate for variations of word embedding size, we
average the word embedding vectors as a dimensionality reduction technique, which performs in our experiments
significantly better than simply padding smaller vectors with 0 or using dimensionality reduction via SVD.

To summarize our changes:

- only one classifier model for all homographs
- average all homograph word embeddings to one feature vector
- additionally use the CLS token as feature vector and combine it with the homograph feature vector

Using this approach, we achieve overall good results in homograph classification for Icelandic when combined with
the **ConvBert** or **LaBSE** models and for those homograph sets that are sufficiently big and balanced.


## Transformer models

The following transformer models were initially examined for generation of the embeddings:

- [distilbert-base-uncased](https://huggingface.co/bert-base-multilingual-cased)
- [icebert](https://huggingface.co/mideind/IceBERT)
- [icebert-large](https://huggingface.co/mideind/IceBERT-large)
- [icelandic-ner-bert](https://huggingface.co/grammatek/icelandic-ner-bert)
- [labse](https://huggingface.co/setu4993/LaBSE)
- [macocu-is](https://huggingface.co/MaCoCu/XLMR-MaCoCu-is)
- [macocu-base-is](https://huggingface.co/MaCoCu/XLMR-base-MaCoCu-is)
- [sbert-ruquad](https://huggingface.co/language-and-voice-lab/sbert-ruquad)
- [convbert-is](https://huggingface.co/jonfd/convbert-base-igc-is)

Due to our reliance on a Wordpiece Tokenizer, the RoBERTa-based models `macocu-is`, `macocu-base-is`, `IceBert`, and
`IceBert-large` are not suitable for generating homograph embeddings, as they utilize a BPE (Byte Pair Encoding)
Tokenizer. Consequently, these models were evaluated solely on their performance using CLS-token embeddings for
classification, which, according to our experiments, is not adequate.
However, it's likely that these models would perform exceptionally well if a method for identifying word embeddings
were developed for BPE-based tokenizers.

## Training set, Training approach and performance measurement

We are using a manually labelled dataset with 73 homograph word forms, generated from the Icelandic Gigaword Corpus
(IGC). The training set is made up of CSV files with 2 columns: a sentence from the IGC containing the homograph marked
via `[[<homograph>]]` and a manually attached label `0`/`1` according to the 2 possible pronunciations of the homograph,
separated by comma. The training set can be retrieved via [Clarin-IS Link](http://hdl.handle.net/20.500.12537/327)

This dataset is highly unbalanced. Therefore, training is done only on the same amount of `1` labels as there are `0`
labels by sampling the same amount of labels. This reduced dataset is split 8-1-1 into train/validation/test set. Due
to balancing and small amount of some homograph labels, we have skipped the following homographs:

- "böllum", "gallanna", "gallarnir", "göllunum", "gella", "halló", "möllum", "pollanna", "villanna"

Leaving us with 64 homographs for training.

All training code can be found in the file [hg_train.py](hg_train.py). The dataset is prepared according to the given
`--around` parameter, then tokenized, embeddings are generated via the given BERT model from the tokenized text and
finally the CLS and homograph word embeddings are isolated to build combined classification features.

As tokenization takes a lot of time, the results are cached into a file alongside the dataset directory, which is
automatically loaded if it exists at training start. This file's name contains the BERT model name as well as the given
value for `--around` for taking into account the distinct parameters. If you don't want to load this cached file and
instead want the training script to recalculate the tokens and embeddings, add `--force` as a parameter to the training
script.

For the classification process, the following parameters can be set. The defaults are shown in parentheses and have
proven to give good results:

- `BATCH_SIZE` (512)
- `N_EPOCHS` (600)
- `ALPHA` (0.00006)
- `TEST_SIZE` (0.1)
- `VALIDATION_SIZE` (0.1)

The accuracy score is calculated at the end for the balanced and unbalanced test sets.

Training with the above settings needs around 16GB VRAM memory on one GPU, additionally to the transformer model usage.
Training on CPU is possible but prohibitively slow (with 32 cores - 20x slower than 1 GPU), because of the BERT
embeddings generation. Training of the classifier itself with the pre-generated data is done on CPU and relatively fast.

## Classifier model

We use SKlearn [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
with the loss function `loss='log_loss'`, i.e. logistic regression. The classifier is saved as a pickle file and the
resulting model file size is just ~7KB. Of course, one always needs additionally the corresponding BERT model for
inference as well.

## Performance

The embeddings generated by the BERT models have proven to make a big difference for the performance of the
classifier. The choice of the right transformer model influences the outcome much more than adjusting the context size
around the homograph from e.g. 5 to 12 or simple hyperparameter tunings.

The model `distilbert-base-uncased` was used as a reference point for a model that has probably not seen any Icelandic
text, to get a feeling for a low baseline. Besides `LaBSE`, which is a multilingual model, the other transformer models
were specifically targeted for Icelandic. As mentioned above, the RoBERTa-based models could only be evaluated on their
CLS-token embeddings and therefore have not been used for the final classification.

The following table shows the accuracy for the different models over the balanced and unbalanced test sets and the
corresponding `--around` parameter with the best result. Each accuracy value averaged over 5 runs.
Best performer is marked in bold, the second best in italic; results are sorted in descending order:

| Model                   | Accuracy balanced | Accuracy unbalanced | --around |
|-------------------------|-------------------|---------------------|----------|
| ConvBert                | **0.9477**        | **0.9239**          | 8        |
| LaBSE                   | *0.9339*          | *0.9176*            | 10       |
| Icelandic-ner-bert      | 0.8870            | 0.8367              | 10       |
| sbert-ruquad            | 0.8708            | 0.8272              | 10       |
| distilbert-base-uncased | 0.7701            | 0.6974              | 10       |

In our experiments, by using only one classifier for all homographs, CLS and word embeddings are both needed to achieve
the above results, though most of these are determined by the homograph word embedding features. Results for single
homographs are omitted here, but those suggest that adding more training data for the less frequent homographs would
improve the overall performance considerably. This is also necessary to be able to classify all homographs, as some
homographs are not classified at all due to lack of training data.

ConvBert and LaBSE prove to be solid choices for Icelandic homograph classification, where the former performs best
in our experiments and also consumes much less GPU memory than the latter.

Our best pretrained model with an accuracy of `0.927` on the unbalanced test set can be found in the directory
`classifier/` for the ConvBert model with parameter `--around 8`. In the same directory, the file `clf_results.csv`
contains detailed training statistics for each homograph.

## Model training

### Prerequisites

Install all dependencies via

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

We have successfully used Python versions 3.9 and 3.10.

### Training

For training of the classifier, the Python script `hg_train.py` is used. It can be run with the following command line

```bash
python3 hg_train.py --directory training_data/ --model convbert --gpu --around 10\
            --output classifier/trained_clf_convbert_10
```

The parameter values for training script stand for:

- directory with training data in path `training_data/`, each homograph in a separate CSV/text file with the format
  `sentence\tlabel`. Labels are marked as (0/1), corresponding to the two possible pronunciations of the homographs 'l'
  and 'tl'
- transformer model: `convbert`
- use GPU
- word context size: `10` tokens (counted via String split !) left and right from the found homograph. The token number
  might be less per direction in case the available number of tokens is less

After training is finished, you can find the classifier model inside the directory `classifier/trained_clf_convbert_10`.

## Model inference

To inference the trained classifier, use the Python script `hg_classify.py`. Please remember to always combine the
correct classifier trained with the specific BERT model !

```bash
python3 hg_classify.py --model convbert --classifier classifier/trained_clf_convbert_10_0.950.pkl -s "Þeir \
     náttúrulega voru í góðum málum , þeir voru búnir að galla sig upp og voru tilbúnir "
```

You can als add the parameter `--gpu` to let it run on your GPU, if available. Via passing the parameter `--file`, you
can classify each line of a file. Results are printed on `stdout`.

# Copyright, Citation

Copyright (C) 2024, Grammatek ehf, licensed via APACHE License v2

If you base any of your research or software on this repository, please consider citing.

```
@misc{IceHoc,
	author={D. Schnell, A.B. Nikulásdóttir},
	title={IceHoc},
	year={2024},
	url={https://www.github.com/grammatek/IceHoc},
}
```
