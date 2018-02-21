#VEP_TMScripts
This repository contains scripts for creating topic models in a directory structure that can be read by [SerendipSlim](https://github.com/uwgraphics/SerendipSlim), a visual model exploration tool. These scripts are somewhat brittle and are provided **AS IS**. They may need to be adjusted or tuned for use on some machines or in certain environments, or to get specific modeling results.

There are two main scripts for creating topic models: **MalletModeler.py** and **GensimModeler.py**.

*Note*: VEP_TMScripts are written in [Python](https://www.python.org/), specifically [version 2.7](https://docs.python.org/2.7/). Python 2.7 must therefore be installed to operate them. We suggest using a Python installation like [Anaconda](https://www.continuum.io/downloads), as this comes with the necessary additional libraries required by our scripts. These include:

- [Gensim](https://radimrehurek.com/gensim/)
- [Scipy](https://www.scipy.org/)
- [NLTK](http://www.nltk.org/)
- [unicodecsv](https://github.com/jdunck/python-unicodecsv) (Optional. For special-character-heavy texts.)

##MalletModeler.py
MalletModeler creates topic models using the [Mallet tool](http://mallet.cs.umass.edu/topics.php) developed by David Mimno, and requires extra software to function properly.

###Mallet
To use MalletModeler, Mallet needs to be installed on your machine and accessible from your system path. Installation instructions for Mallet can be found [here](http://mallet.cs.umass.edu/download.php). Instructions for updating your system path environment variable can be found [here for Windows](https://www.java.com/en/download/help/path.xml) and [here for Macs](https://www.cyberciti.biz/faq/appleosx-bash-unix-change-set-path-environment-variable/).

> *Note*: In the past, I have typically used Mallet version 2.0.7. There may be some issues associated with using their new version, 2.0.8.

###Running MalletModeler
Once the required third-party software has been properly installed, MalletModeler can be run from within the VEP_TMScripts directory with the simple command python MalletModeler.py. However, this will not create a new topic model unless provided with the proper arguments. MalletModeler requires the following arguments:

- `--corpus_path` (REQUIRED): A path specifying a directory containing the .txt files that will form the basis for the topic model being created. This directory should only contain .txt files.
- `--output_path` (REQUIRED): A path specifying a directory in which the topic model will be written. MalletModeler will create a new directory within this directory named with the model_name, in which will be stored the files and directories needed by SerendipSlim.
- `--model_name` (REQUIRED): A name for the specific model being created. MalletModeler will create a directory by this name within the output_path directory and use it to store the files and directories needed by SerendipSlim.
- `--num_topics` (REQUIRED): The number of topics that should be extracted from the documents within the corpus.

So, for instance, a model could be created using a command such as:

```
> python MalletModeler.py --corpus_path C:\Corpora\Shakespeare --output_path C:\SerendipModels --model_name Shake_Mallet_50 --num_topics 20
```

MalletModeler can take other arguments that can give researchers greater control over the models they build:

- `--chunk_size`: MalletModeler can divide documents into chunks of fewer tokens before building the topic model. This practice of "chunking" restricts the definition of what it means for two words to appear "together," and can help create finer-grained models that show greater variation of topics within documents. By default, chunking is disabled. Others have shown that chunk sizes of around 1000 tokens can be effective (https://tedunderwood.com/2012/03/25/a-touching-detail-produced-by-lda/).
- `--num_iterations`: The number of iterations Mallet will perform when inferring the topics. More iterations will give the model more time to converge on stable topics. (default: 100)
- `--mallet_stopwords`: This argument does not require an additional value. Rather, by including the --mallet_stopwords flag, a researcher can have MalletModeler extract the standard set of English stopwords included in Mallet before building the topic model. The practice of excluding stopwords can keep models from being dominated by very common and uninteresting words.
- `--extra_stopword_path`: A path specifying a .txt file containing space-delimited stopwords that the researcher wants to exclude from the model. This argument can be used in conjunction with the --mallet_stopwords flag to combine the two sets of stopwords, or it can be used by itself (e.g., if the documents have an unusual set of uninformative words).

Additional arguments (related to [Mallet's command line arguments](http://mallet.cs.umass.edu/topics.php)) can be found with the command `python MalletModeler.py --help`

If everything works properly, the files SerendipSlim needs to display the newly created topic model will be created in a new directory within the given output_path by the name of model_name. For instructions on how to point a local installation of SerendipSlim at this new model, see the [SerendipSlim README](https://github.com/uwgraphics/SerendipSlim#serendipslim-model-format).

##GensimModeler.py
GensimModeler creates topic models using the Python library [Gensim](https://radimrehurek.com/gensim/). Gensim has grown in popularity in particular thanks to its implementation of deep learning in the popular [word2vec](https://radimrehurek.com/gensim/models/word2vec.html), but also is equipped to create compact topic models over streamed text data using LDA. To be overly honest, GensimModeler does not take advantage of Gensim's real strengths (e.g., being able to create models of extremely large corpora using a streamed algorithm), and primarily provides a (brittle) way of hacking Gensim output into something like what SerendipSlim requires. My use of Gensim's modeling parameters leaves something to be desired, and needs to be improved. Nonetheless, here it is!

###Running GensimModeler
Like MalletModeler, GensimModeler can be run from within the VEP_TMScripts directory with the simple command python GensimModeler.py. However, this will not create a new topic model unless provided with the proper arguments. GensimModeler requires the following arguments:

- `--corpus_path` (REQUIRED): A path specifying a directory containing the .txt files that will form the basis for the topic model being created. This directory should only contain .txt files.
- `--output_path` (REQUIRED): A path specifying a directory in which the topic model will be written. MalletModeler will create a new directory within this directory named with the model_name, in which will be stored the files and directories needed by SerendipSlim.
- `--model_name` (REQUIRED): A name for the specific model being created. MalletModeler will create a directory by this name within the output_path directory and use it to store the files and directories needed by SerendipSlim.
- `--num_topics` (REQUIRED): The number of topics that should be extracted from the documents within the corpus.

So, for instance, a model could be created using a command such as:

```
> python GensimModeler.py --corpus_path C:\Corpora\Shakespeare --output_path C:\SerendipModels --model_name Shake_Gensim_50 --num_topics 20
```

GensimModeler can take other arguments that can give researchers greater control over the models they build:

- `--chunk_size`: Like MalletModeler, GensimModeler can divide documents into chunks of fewer tokens before building the topic model, potentially increasing the topic variance within documents. I have not achieved fantastic results when chunking using Gensim--in general, I tend to get chunks dominated by one (or at most two) topics--but it is likely to be improve-able with some tweaking of the model parameters. By default, chunking is disabled.
- `--num_iterations`: The number of iterations Gensim will perform when inferring the topics. More iterations will give the model more time to converge on stable topics. (default: 100)
- `--nltk_stopwords`: This argument does not require an additional value. Rather, by including the --mallet_stopwords flag, a researcher can have GensimModeler extract the standard set of English stopwords included in the [NLTK Python library](http://www.nltk.org/) before building the topic model. The practice of excluding stopwords can keep models from being dominated by very common and uninteresting words.
- `--extra_stopwords_path`: A path specifying a .txt file containing space-delimited stopwords that the researcher wants to exclude from the model. This argument can be used in conjunction with the --mallet_stopwords flag to combine the two sets of stopwords, or it can be used by itself (e.g., if the documents have an unusual set of uninformative words).

As with MalletModeler, if everything works properly, the files SerendipSlim needs to display the newly created topic model will be created in a new directory within the given output_path by the name of model_name. For instructions on how to point a local installation of SerendipSlim at this new model, see the [SerendipSlim README](https://github.com/uwgraphics/SerendipSlim#serendipslim-model-format).

##Note of warning, and an entreaty:
Again, these scripts were written to be used by a handful of people on a handful of machines--they have not been tested within a variety of environments, and they do not take full advantage of the modeling capabilities of either Mallet or Gensim. They may be updated over time, but if you are interested in building better models, please feel free to improve them as you see fit!
