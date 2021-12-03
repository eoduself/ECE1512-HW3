There are five py files where **config.py / conventional_aug.py** is for **conventional augmentations** and **config_GAN.py / GAN_model.py / pre_processing_GAN.py** is for **GAN part**. In addition, I shared the sample data (French text) for running those codes in *s3://bluetrain-eu-workspaces/dyhwang/CR_data*. 

## 1. config.py
There are five parameters in this file. <br />
(1) INPUT_NAME: It is for original text data. Now, it can receive npy file.<br />
(2) AUG_METHOD: It is for choosing augmentation methods. There are four methods: word-level / character-level / translation / back-translation.<br />
(3) WRD_SPECIFIC: It is for selecting lexical or spelling-based augmentation in word-level. It works when AUG_METHOD=’word’.<br />
(4) CHR_SPECIFIC: It is for choosing insert, substitute, swap or delete method in character-level. It works when AUG_METHOD=’character’. <br />
(5) OUTPUT_NAME: It denotes the name of output file which contains the synthetic data.<br />

## 2. conventional_aug.py
This contains the four augmentation methods: word-level / character-level / translation / back-translation. I add the Prerequisites in each method but you can also check them at below. **If you don’t want to use some approaches, please eliminate those import parts.** It receives the input text file (npy) and generate the output text file (npy). You can use the sample_example.npy in the bucket for sample run. <br />
**To run the code:** python conventional_aug.py <br />
(1) Word-level: For lexical-based, it is basically finding a synonym from WordNet and for spelling-based, it is changing some characters for synthesizing. For both methods, you can change the amount of augmentation with aug_p in function. <br />
(2) Character-level: Four different character augmentations can be considered where you can change the amount of augmentation with aug_char_p in function. <br />
(3) Translation: Now, it is based on De -> En -> Fr. Please use the sample_example_de.npy in INPUT_NAME of config for sample run. <br />
(4) Back-Translation: Now, it is based on Fr -> En -> Fr. <br />
**Prerequisites** <br />
Word-level: nlpaug-1.1.7/ nltk: averaged_perceptron_tagger / nltk: wordnet / nltk: omw <br />
*https://www.nltk.org/nltk_data/* <br />
Character-level: nlpaug-1.1.7 <br />
Translation: EasyNMT-2.0.1.tar / nltk: punkt / pre-trained mt from *https://huggingface.co/Helsinki-NLP* <br />
*https://github.com/UKPLab/EasyNMT* <br />
Back-Translation: Same as Translation <br />
 
## 3. config_GAN.py
There are ten parameters in this file. <br />
(1) INPUT_NAME: It is for original text data. Now, it can receive npy file. <br />
(2) GAN_METHOD: It is for choosing GAN approaches. There are three methods: FastText / BERT / BART. <br />
(3) DICT_NAME: It is dictionary for comparison in BERT-GAN. The sample (dict_embedding_bert_gan.npz) comes from translated German texts.  <br />
(4) BATCH_SIZE: It is batch size used in GAN. <br />
(5) EPOCHS: The number of epochs in GAN. <br />
(6) LEARNING_RATE: Learning rate in GAN. <br />
(7) FASTTEXT_SCORE: Threshold for measuring similarity in FastText-GAN. <br />
(8) BART_SCORE_LOW: Low threshold for measuring similarity in BART-GAN <br />
(9) BART_SCORE_HIGH: High threshold for measuring similarity in BART-GAN <br />
(10) OUTPUT_NAME: It denotes the name of output file which contains the original text and its embedding together. This is also used in GAN as train set. <br />

## 4. pre_processing_GAN.py 
This is for changing the text into embedding before applying in the GAN model. Three methods are considered: FastText-GAN / BERT-GAN / BART-GAN. **If you don’t want to use some approaches, please eliminate those import parts.** It receives the text file (npy) and generate the embedding file (npz) with its original text. You can use the sample_example_GAN.npy in the bucket for sample run. <br />
**To run the code:** python pre_processing_GAN.py <br />
(1) FastText-GAN: In this model, synthetic data is generated for each single word and thus, normalization task is applied. The process is as follows: Remove stop words -> Remove numbers -> Remove punctuation -> Split data according to whitespace. Then, pre-trained FastText is used to generate the embedding for each single word where the default size is 300. <br />
(2) BERT-GAN: In this model, synthetic data is generated for each sentence. To get the embedding of each sentence, I average the second to last hidden layer of each token of sentence, and also average the tokens in a sentence. Finally, this gives 768 hidden units for each sentence. <br />
(3) BART-GAN: In this model, synthetic data is generated for each sentence. I only consider the sentence less than 7 tokens since most sentences satisfy this condition. Here, each data consists of 768 x 6 since each token has 768 hidden units from encoder of BART and occupies each dimension. <br /> 
**Prerequisites** <br />
FastText-GAN: fasttext / stop-words-2018.7.23.tar/ pretrained-model from *https://fasttext.cc/docs/en/crawl-vectors.html* <br />
BERT-GAN: pretrained-model from *https://github.com/google-research/bert/blob/master/multilingual.md* <br />
BART-GAN: recent transformer (ex.4.12.5) / pretrained-model from *https://github.com/moussaKam/BARThez* <br />

## 5. GAN_model.py
This is for training the GAN model and generate the synthetic data at last. Three methods are considered: FastText-GAN / BERT-GAN / BART-GAN. **If you don’t want to use some approaches, please eliminate those import parts.** It receives the embedding file (npz) and generates the synthetic text file for each original text (npy). Output file has dictionary type (i.e. original text: synthetic text1, synthetic text2 …). You need to run pre_processing_GAN.py first since GAN needs the embedding of original texts.<br />
Generator in FastText is slightly different from BERT and BART since the size of data is different (300 vs 768). Discriminator is same for all approaches. Training checkpoints are saved and thus, you can call the model later. After finishing the GAN training, the loss function is saved and different methods are used to generate synthetic data in each GAN method. For reference, the sample example is not enough for getting proper synthetic data (in this case, some original text does not have synthetic data) but you can see how the code works. The time complexity of code can be decreased if we split the GAN training and generation of synthetic data since the latter one can be done parallel. <br />
**To run the code:** python GAN_model.py <br />
(1) FastText-GAN: Embedding of original text with Gaussian noise is applied in Generator to produce the synthetic embedding. Then, cosine similarity between synthetic embedding and embeddings in dictionary of pre-trained FastText is applied to find the most similar words. Finally, additional similarity matching between selected words from dictionary and original word is done for finding most proper synthetic word. <br />
(2) BERT-GAN: Embedding of original text with Gaussian noise is applied in Generator to produce the synthetic embedding. Then, cosine similarity between synthetic embedding and embeddings from dictionary is applied to find the most similar sentences. The dictionary comes from translation of other language (the example comes from German).  <br />
(3) BART-GAN: Embedding of original text with Gaussian noise is applied in Generator to produce the synthetic embedding. Then, original and synthetic embeddings are averaged. In each dimension, cosine similarity between synthetic embedding and embeddings from dictionary is applied. Finally, decoder in BART is applied to make synthetic data.<br />
**Prerequisites are same as pre_processing_GAN.py**<br />



 

