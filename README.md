# Deep-BGCpred: A unified deep learning genome-mining framework for biosynthetic gene cluster prediction

Deep-BGCpred, a framework that effectively addresses the aforementioned issue by improving a deep learning model termed DeepBGC. The new model embeds multi-source protein family domains and employs a stacked Bidirectional Long Short-Term Memory model to boost accuracy for BGC identifications. In particular, it integrates two customized strategies, sliding window strategy and dual-model serial screening, to improve the model’s performance stability and reduce the number of false positive in BGC predictions.



![](https://github.com/yangziyi1990/Deep-BGCpred/blob/main/images/Figure1_pipeline.png)

If you find this code useful in your research, then please cite:

```
@article{
  title={Deep-BGCpred: A unified deep learning genome-mining framework for biosynthetic gene cluster prediction},
  author={Yang, Ziyi, Liao Benben, Hsieh Changyu, Han Chao, Fang Liang, Zhang Shengyu},
  year={2021},
}
```



## Installation

Follow the steps in the defined order to avoid conflicts.

1. The models.zip file will have to be downloaded and unzipping.

   ```
   $ git clone https://github.com/pmobio/Deep-BGCpred.git
   ```

2. Create an environment:

   ```
   # install the DeepBGCpred environment
   $ conda env create -f requirements_env.yml
   
   # To activate this environment, use
   $ conda activate DeepBGCpred
   
   # To deactivate an active environment, use
   $ conda deactivate
   ```

3. Install external packages:

   - Install HMMER and put it on your PATH: http://hmmer.org/download.html 

     ```
     # Install HMMER
     $ wget http://eddylab.org/software/hmmer/hmmer.tar.gz
     $ tar zxf hmmer.tar.gz
     $ cd hmmer-3.3.2
     $ ./configure --prefix /your/install/path
     $ make
     $ make check                 
     $ make install
     
     # add the HMMER path to your PATH (/your/install/path)
     $ vim ~/.bashrc
     $ export PATH=/your/install/path/hmmer-3.3.2/bin:$PATH
     $ source ~/.bashrc
     ```

   - Install Prodigal and put the prodigal binary it on your PATH: https://github.com/hyattpd/Prodigal/releases

     ```
     $ conda install -c bioconda prodigal
     # or 
     $ tar -xzvf Prodigal-2.6.3.tar.gz
     $ make install
     ```



## Usage

First, you need access the work path

```
$ cd deepbgcpred
```



### Download the models and the Pfam database

Before you can use DeepBGCpred, download trained models and Pfam database:

```
$ python main.py download
```

After downloaded the dependencies and models, you can check them:

```
$ python main.py info
```



### Prepare the training data

If you want to training the Deep-BGCpred model on you own data, you need to prepare the training data firstly. 

```
# See the usage of generate the Pfam TSV format file from FASTA sequence
$ python main.py prepare --help
```

In the Deep-BGCpred model, we achieved three functions for the training data preparation: 

1) The training data need to be prepared in Pfam TSV format, which can be prepared from your sequence using:

   ```
   $ python main.py prepare --pre --inputs sequence.fa --output-tsv sequence.prepared.tsv
   ```

2) Add the Clan and description information to the Pfam TSV file; 

   ```
   $ python main.py prepare --add --clan-txt Pfam-A.clans.tsv --prepared-tsv sequence.prepared.tsv --output-new-tsv sequence.prepared.new.tsv
   ```

3) Data augmentation according to the Pfam TSV file.

   ```
   $ python main.py prepare --aug --interaction-txt pfamA_interactions.txt --pfam-tsv sequence.prepared.new.tsv -r 0.02 -n 2 --output-aug-tsv sequence.prepared.aug.tsv
   ```



### BGC prediction and classification

#### 1. Check your training dataset

Prepare the training dataset to train the models for BGC prediction and classification.  Training and validation data can be found in https://github.com/pmobio/Deep-BGCpred/releases/tag/v0.1.0.

- Positive (BGC) training data provided by DeepBGC [1]. If you want to generate your own BGC training dataset, please see "Prepare the training data" section mentioned above.
- Negative (Non-BGC) training data. Your can use `GeneSwap_Negatives.pfam.new.tsv` as the negative samples to train the predition model. For the classification stage,  the negative samples are taken from two sources: data generation based on the negative samples released by [1] (2000 samples), and the non-BGC samples incorrectly predicted by Bi-LSTM network in the preceding stage of the training process (102 samples). 

- Validation dataset [1, 2]: Needed for BGC detection. Contigs with annotated BGC and non-BGC regions, see https://github.com/pmobio/Deep-BGCpred/releases/tag/v0.1.0.
- Trained Pfam2vec vectors provided by DeepBGC [1]: "Vocabulary" converting Pfam IDs to meaningful numeric vectors, results from https://github.com/Merck/deepbgc/releases/tag/v0.1.0.
- Annotated Clan and description information: Record the Pfam, Clan and description information. 
- Pfam-Pfam interaction file. Record the interaction information of Pfam-Pfam interaction (PPI).
- BGC.Classes.csv: Chemical product class used to train the random forest model.
- JSON configuration files: See JSON section below.

#### 2. JSON configuration file

Deep-BGCpred is using JSON template files to define model architecture and training parameters, which is the same with DeepBGC

```
{
  "type": "KerasRNN",
  "build_params": {        - Parameters for model architecture
    "batch_size": 64,      - Number of splits of training data that is trained in parallel 
    "hidden_size": 128,    - Size of vector storing the LSTM inner state
    "stateful": true       - Remember previous sequence when training next batch
  },
  "fit_params": {
    "timesteps": 256,             - Number of pfam2vec vectors trained in one batch
    "validation_size": 0.1,       - Fraction of training data to use for validation (if validation data is not provided explicitly). Use 0.2 for 20% data used for testing.
    "verbose": 1,
    "num_epochs": 1000,           - Number of passes over your training set during training. You probably want to use a lower number if not using early stopping on validation data.
    "early_stopping" : {          - Stop model training when at certain validation performance
      "monitor": "val_auc_roc",   - Use validation AUC ROC to observe performance
      "min_delta": 0.0001,        - Stop training when the improvement in the last epochs did not improve more than 0.0001
      "patience": 20,             - How many of the last epochs to check for improvement
      "mode": "max"               - Stop training when given metric stops increasing (use "min" for decreasing metrics like loss)
    },
    "shuffle": true,              - Shuffle samples in each epoch. Will use "sequence_id" field to group pfam vectors belonging to the same sample and shuffle them together 
    "optimizer": "adam",          - Optimizer algorithm
    "learning_rate": 0.0001,      - Learning rate
    "weighted": true              - "learning_rate": 0.0001,  - Learning rate


  },
  "input_params": {
    "features": [
      {
        "type": "ProteinBorderTransformer"             - Add two binary flags for pfam domains found at beginning or at end of protein
      },
      {
        "type": "Pfam2VecTransformer",                 - Convert pfam_id field to pfam2vec vector using provided pfam2vec table
        "vector_path": "#{PFAM2VEC}"                   - PFAM2VEC variable is filled in using command line argument --config
      },
      {
        "type": "RandomVecTransformerforClan"          - Add Clan to the features  
      },
      {
        "type": "ColumnSelectTransformerDescription"   - Add Description information to the features
      }
    ]
  }
}
```

JSON template for Random Forest classifier is structured as follows, which is the same with the work in [1]:

```
{
  "type": "RandomForestClassifier",     - Type of classifier (RandomForestClassifier)
  "build_params": {
    "n_estimators": 100,                - Number of trees in random forest
    "random_state": 0                   - Random seed used to get same result each time
  },
  "input_params": {
    "sequence_as_vector": true,         - Convert each sample into a single vector
    "features": [
      {
        "type": "OneHotEncodingTransformer" - Convert each sequence of Pfams into a single binary vector (Pfam set)
      }
    ]
  }
}
```



#### 3. Train the BGC prediction and classification models

Before predict the BGCs and classify the class of BGCs, you need to train your own BGC detector and classifier.

```
# Show train command help docs
$ python main.py train --help

# Train a detector using pre-processed samples in Pfam CSV format. 
$ python main.py train --model deepbgcpred.json --output MyDeepBGCpredDetector.pkl BGCs.pfam.tsv negatives.pfam.tsv

# If you are training a DeepBGC detector, you need to add the parameter -i
$ python main.py train --model deepbgc.json -i 102 --output MyDeepBGCpredDetector.pkl BGCs.pfam.tsv negatives.pfam.tsv

# Train a BGC classifier using a TSV classes file and a set of BGC samples in Pfam TSV format and save the trained classifier to a file.
$ python main.py train --model random_forest.json --output MyDeepBGCpredClassifier.pkl --classes BGCs.classes.csv BGCs.pfam.tsv
```



#### 4. BGC prediction and classification

In this step, the proteins and Pfam domains are detected automatically if not already annotated (the same with the work in [1] using HMMER and Prodigal).

```
# Show pipeline command help docs
$ python main.py pipeline --help

# Detect and classify BGCs in mySequence.fa using the DeepBGCpred detector without sliding window strategy
$ python main.py pipeline mySequence.fa --pcfile --pfam-clain-file Pfam-A.clans.tsv --detector myDetector.pkl --classifier myClassifier.pkl 

# Detect and classify BGCs in mySequence.fa using the DeepBGCpred detector with sliding window strategy
$ python main.py pipeline mySequence.fa --pcfile --pfam-clain-file Pfam-A.clans.tsv --detector myDetector.pkl --classifier myClassifier.pkl --sliding-window -sw_width 256 -sw_steps 20

# Detect and classify BGCs in mySequence.fa using the DeepBGCpred detector with sliding window strategy and dual-model serial screening
$ python main.py pipeline mySequence.fa --pcfile --pfam-clain-file Pfam-A.clans.tsv --detector myDetector.pkl --classifier myClassifier.pkl --sliding-window -sw_width 256 -sw_steps 20 --screening

# If you want to detect and classify BGCs in mySequence.fa using the DeepBGC detector, please add a parameters -i
$ python main.py pipeline mySequence.fa -i 102 --detector myDetector.pkl --classifier myClassifier.pkl
```

This will produce a mySequence directory with multiple files and a README.txt with file descriptions.

Example output:

![](https://github.com/yangziyi1990/Deep-BGCpred/blob/main/images/Figure8_genome_anno.png)



# Reference

[1] Hannigan, Geoffrey D., et al. "A deep learning genome-mining strategy for biosynthetic gene cluster prediction." Nucleic acids research 47.18 (2019): e110-e110.

[2] Peter Cimermancic, Marnix H Medema, Jan Claesen, Kenji Kurita, Laura C Wieland Brown, Konstantinos Mavrommatis, Amrita Pati, Paul A Godfrey, Michael Koehrsen, Jon Clardy, et al. Insights into secondary metabolism from a global analysis of prokaryotic biosynthetic gene clusters. Cell, 158(2):412–421, 2014.
