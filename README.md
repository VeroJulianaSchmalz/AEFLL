# AEFLL (Automatic Evaluation of Foreign Language Learning): _An interdisciplinary project for English, Italian and German at the Free University of Bozen-Bolzano_ 


This repository contains a master's thesis study focused on the **automatic assessment of written and oral language competences** of _adult_ learners of _English, German, and Italian_ using **BERT models** (available at https://huggingface.co/transformers/pretrained_models.html). Given the exponential growth of the latter worldwide and the increasing adoption of computer-assisted language examinations, these automated systems could facilitate the _objective_ scrutiny of numerous tests, _reducing the biases_ of human evaluators while providing cross-validly efficient and detailed assessments. We combine analysis and assessment methods within machine learning, natural language processing, language acquisition and development to correct and classify both written and oral examinations of adult language learners following the principles of the Common European Framework of Competence. 
For our analysis we use written open-source datasets of English proficiency tests, namely EFCAMDAT[^1] and CLC-FCE[^2], and MERLIN[^3] for Italian and German. To train our models (see Training below), we conduct different experiments alternatively using the original written texts from the students, human corrections, when available, and the automatic corrections provided by LanguageTool (https://github.com/languagetool-org/languagetool), a computerized language checker tool. In this way the BERT model provides an embedding representation that not only describes the text content but at times also accounts for rule violations and other errors. A multi-layer perceptron is then used to map the embedding text representation into the related CEFR levels. We evaluate the performance of our architecture on each dataset training a language specific model, achieving extremely high proficiency prediction in all cases. In addition, we received a narrow dataset of oral examinations for B2 English exams from the Free University of Bozen's Language Centre[^4] on which we conduct a separate case study. We applied the pretrained English models on the oral exams, previously transcribed using an automatic speech recognition engine. Finally, we consider linguistic aspects related to the written and spoken language of the learners of different languages and possible features to be added to the models to possibly improve their performance.

## Datasets 

- **English First Cambridge Open Language Database (EFCAMDAT)**:  1,180,310 texts, from A1 to C2 level
- **Cambridge Learner Corpus for the First Certificate in English exam(CLC- FCE)**: 2,469 texts, between A2 and B2 levels 
- **MERLIN Italian**: 813 texts, from A1 to B2 level
- **MERLIN German**: 1,033 texts, from A1 to C2 level 
- **UNIBZ exams** : 60 oral exams for the English B2 level 


## Pre-processing 

Once we obtained the different corpora, we organized the learners' original texts and other related metadata, like student's first language and original text, number of tokens in text, attested level of competence, corrections and found errors, into _.tsv_ files, the content of which resulted similar to the table below:

| Student_ID  | L1  | Text | Tokens  | CEFR_level  |
| ------------ |---------------| -----| -----| -----|
| 1091_0000055     |Russian | Liebe Maria, wie gehts dir? Mir geht gut, ich...	 |150| B1|

To this data, with the use of LanguageTool, we added the type and number of errors systematically detected, and automatic corrections, which we compared with the manual ones made occasionally available. 

```
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
 
text = """Liebe Maria, wie gehts dir? Mir geht gut, ich..."""
 
# get the matches
matches = tool.check(text)
matches.keys() 

--> PUNKT_NACH_ORDINALZAHL, DE_VERBAGREEMENT, KOMMA_ZWISCHEN_HAUPT_UND_NEBENSATZ...

tool.correct(text)

--> """Liebe Maria, wie gehts dir? Mir geht es gut, ich..."""
```

## Initial errors and levels analysis

Before adopting the BERT-base model, we also considered and compared the amount and type of errors found by human and the automatic tool. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/72256200/136711936-23042b76-95b6-4642-88ba-1d7162c1ab27.png">
</p>

<h5 align="center">
Comparing errors' quantities across levels and correction modalities in EFCAMDAT</h5>



<p align="center">
  <img src="https://user-images.githubusercontent.com/72256200/136711961-ab7bd326-1be9-4b91-a4e7-9405e314e7e4.png" alt="_Comparing error types across different levels of the EFCAMDAT corpus_" >
</p>

<h5 align="center">
Comparing error types across different levels of the EFCAMDAT corpus</h5>



## Linguistic analysis of learners' data 

We analyzed both the lexical richness of the learners’ texts, applying TTR, HD-D and MTLD measures, and the syntactic complexity, considering the average sentence length in words and the dependency distance, together with the number of dependents per word unit. After having tokenized, lemmatized and parsed the texts, we stored the results using the CoNLL-U format. These originated files were fed to _textcomplexity_ (available at https://github.com/tsproisl/textcomplexity). 
We expected to observe possible correlations between the increasing levels of competence and the possibly increasing text complexity features. 


## BERT-based model

For our experiments in multi-class classification we used a pre-trained model, _BERT-base-uncased_. The main architecture consisted of: 

- A Tokenizer, which translates raw text strings into sparse index encodings; 
- A Transformer, which transforms the previously generated sparse indices into contextual embeddings;
- A fixed Head, in our case BERT, which uses contextual embeddings to generate the specific predictions for text classification task.


<p align="center">
  <img src="https://user-images.githubusercontent.com/72256200/136712498-80d54851-2fb3-4dc7-b0a9-a5c5d11dc56a.png">
</p>





The goal of the model is classifying the original language learners' texts into one of the 5 given classes, which correspond to one of the CEFR scale levels. We conducted several experiments, some using only the original student submitted texts (with the left model), and some additionally providing the model either with human or automatic corrections (with the right model). In the case of the oral exams, the model received ASR-generated (and manually checked) transcriptions. 

## Results 

The obtained results across the three different languages proved the efficiency of our system in identifying the CEFR levels of the learners' texts, reaching a major accuracy of 92,2%. However, due to the larger amount of data available for English, in that language we were able to obtain the most outstanding results. The use of an automatic system by which to systematically correct learners' exams and detect errors would seem to improve the performance of our automatic system to some extent. Especially with regard to oral examinations, we found a possible _bias effect_ in the assessment of competences. Nevertheless, several questions remain open in this regard, clarifiable perhaps with more data across different levels of competence. Also in the case of Italian and German, there are possible improvements reachable with a larger amount of texts. 







[^1]: https://philarion.mml.cam.ac.uk/ 
[^2]: https://ilexir.co.uk/datasets/index.html
[^3]: https://merlin-platform.eu/
[^4]: https://www.unibz.it/en/services/language-centre/
