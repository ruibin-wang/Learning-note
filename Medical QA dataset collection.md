# Medical QA dataset collection

## MedQA

* Multiple choice question answering based on the United States Medical License Exams (USMLE). The dataset is collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and traditional Chinese, and contains **12,723, 34,251, and 14,123 questions** for the three languages, respectively.

* The link can be find here: https://github.com/jind11/MedQA

* examples:

```json
{
    "question": "A 32-year-old woman comes to her doctor’s office with abdominal distention, diffuse abdominal pain, and a history of 10–12 bowel movements a day for the last week. She was diagnosed with Crohn’s disease 3 years ago. Today, vitals are normal. Her abdomen is mildly distended and diffusely tender to palpation. A CT scan shows evidence of a fistula and strictures located in the last 30 cm of her ileum. A resection of the affected portion of the bowel is scheduled. What changes in bile metabolism are expected in this patient post-procedure?",
    "answer": "The balance of the components in bile will be altered",
    "options": {
        "A": "Absorption of vitamin K will not be impaired",
        "B": "Synthesis of cholesterol in the liver will decrease",
        "C": "The balance of the components in bile will be altered",
        "D": "Enteric bacteria will remain the same in the small intestine",
        "E": "Absorption of  7⍺-dehydroxylated bile will decrease"
    },
    "meta_info": "step1",
    "answer_idx": "C"
}
```

## PubMedQA

* PubMedQA, a novel biomedical question answering (QA) dataset collected from PubMed abstracts. The task of PubMedQA is to answer research questions with yes/no/maybe (e.g.: Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?) using the corresponding abstracts. PubMedQA has 1k expert-annotated, 61.2k unlabeled and 211.3k artificially generated QA instances. Each PubMedQA instance is composed of (1) a question which is either an existing research article title or derived from one, (2) a context which is the corresponding abstract without its conclusion, (3) a long answer, which is the conclusion of the abstract and, presumably, answers the research question, and (4) a yes/no/maybe answer which summarizes the conclusion. PubMedQA is the first QA dataset where reasoning over biomedical research texts, especially their quantitative contents, is required to answer the questions.


* The link can be find here: https://github.com/pubmedqa/pubmedqa

* paper can be find here: https://arxiv.org/pdf/1909.06146.pdf

* examples:

```json
    {
        "25433161": {
        "QUESTION": "Does vagus nerve contribute to the development of steatohepatitis and obesity in phosphatidylethanolamine N-methyltransferase deficient mice?",
        "CONTEXTS": [
            "Phosphatidylethanolamine N-methyltransferase (PEMT), a liver enriched enzyme, is responsible for approximately one third of hepatic phosphatidylcholine biosynthesis. When fed a high-fat diet (HFD), Pemt(-/-) mice are protected from HF-induced obesity; however, they develop steatohepatitis. The vagus nerve relays signals between liver and brain that regulate peripheral adiposity and pancreas function. Here we explore a possible role of the hepatic branch of the vagus nerve in the development of diet induced obesity and steatohepatitis in Pemt(-/-) mice.",
            "8-week old Pemt(-/-) and Pemt(+/+) mice were subjected to hepatic vagotomy (HV) or capsaicin treatment, which selectively disrupts afferent nerves, and were compared to sham-operated or vehicle-treatment, respectively. After surgery, mice were fed a HFD for 10 weeks.",
            "HV abolished the protection against the HFD-induced obesity and glucose intolerance in Pemt(-/-) mice. HV normalized phospholipid content and prevented steatohepatitis in Pemt(-/-) mice. Moreover, HV increased the hepatic anti-inflammatory cytokine interleukin-10, reduced chemokine monocyte chemotactic protein-1 and the ER stress marker C/EBP homologous protein. Furthermore, HV normalized the expression of mitochondrial electron transport chain proteins and of proteins involved in fatty acid synthesis, acetyl-CoA carboxylase and fatty acid synthase in Pemt(-/-) mice. However, disruption of the hepatic afferent vagus nerve by capsaicin failed to reverse either the protection against the HFD-induced obesity or the development of HF-induced steatohepatitis in Pemt(-/-) mice."
        ],
        "LABELS": [
            "OBJECTIVE",
            "METHODS",
            "RESULTS"
        ],
        "LONG_ANSWER": "Neuronal signals via the hepatic vagus nerve contribute to the development of steatohepatitis and protection against obesity in HFD fed Pemt(-/-) mice.",
        "MESHES": [
            "Animals",
            "Chemokine CCL2",
            "Diet, High-Fat",
            "Disease Models, Animal",
            "Fatty Liver",
            "Interleukin-10",
            "Liver",
            "Mice",
            "Obesity",
            "Phosphatidylcholines",
            "Phosphatidylethanolamine N-Methyltransferase",
            "Postoperative Period",
            "Transcription Factor CHOP",
            "Vagotomy",
            "Vagus Nerve"
        ],
        "final_decision": "yes"
        }
    }
```

## MMLU(Measuring Massive Multitask Language Understanding)

* "Measuring Massive Multitask Language Understanding" (MMLU) includes exam questions from 57 domains. We selected the subtasks most relevant to medical knowledge: "anatomy", "clinical knowledge", "college medicine", "medical genetics", "professional medicine", and "college biology". Each MMLU subtask contains multiple-choice questions with four options, along with the answers.


* The link can be find here: https://github.com/hendrycks/test

* paper can be find here: https://arxiv.org/pdf/2009.03300.pdf

* examples:



```js
Format: Q + A, multiple choice, 
open domain AnatomySize (Dev/Test): 14 / 135
Question: Which of the following controls body temperature, sleep, and appetite?
Answer: (A) Adrenal glands (B) Hypothalamus (C) Pancreas (D) Thalamus 

Clinical KnowledgeSize (Dev/Test): 29 / 265
Question: The following are features of Alzheimer's disease except:
Answer: (A) short-term memory loss. (B) confusion. (C) poor attention. (D) drowsiness.
College MedicineSize (Dev/Test): 22 / 173

Question: The main factors determining success in sport are:
Answer: (A) a high energy diet and large appetite. (B) high intelligence and motivation to succeed. (C) a good coach and the motivation to succeed. (D) innate ability and the capacity to respond to the training stimulus.


```


## LiveQA

* A new question answering dataset constructed from play-by-play live broadcast. It contains 117k multiple-choice questions written by human commentators for over 1,670 NBA games, which are collected from the Chinese Hupu (https://nba.hupu.com/games) website.


* The LiveQA dataset was curated as part of the Text Retrieval Challenge (TREC) 2017. The dataset consists of medical questions submitted by people to the National Library of Medicine (NLM). The dataset also consists of manually collected reference answers from trusted sources such as the National Institute of Health (NIH) website.

* The dataset can be found here: https://github.com/PKU-TANGENT/LiveQA




```js
Format: Q + long answers, free text response, open domainSize (Dev/Test): 634/104
Question: Could second hand smoke contribute to or cause early AMD?
Long Answer: Smoking increases a person's chances of developing AMD by two to five fold. Because the retina has a high rate of oxygen consumption, anything that affects oxygen delivery to the retina may affect vision. Smoking causes oxidative damage, which may contribute to the development and progression of this disease. Learn more about why smoking damages the retina, and explore a number of steps you can take to protect your vision.

```




## BIG-bench (Beyond the Imitation Game Benchmark)

* The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to probe large language models and extrapolate their future capabilities. Big-bench include **more than 200 tasks**.

* Image shows a word cloud of task keywords.


* Paper can be found here: https://arxiv.org/pdf/2206.04615.pdf



## dataset from other sources


I) Medical QA Datasets:
=======================

1) Corpus for Evidence Based Medicine Summarization (Mollá, 2010): https://sourceforge.net/projects/ebmsumcorpus 
2) CLEF QA4MRE Alzheimer’s task (Peñas et al, 2012). 
3) BioASK datasets (2012-2020): http://bioasq.org/participate/challenges
4) TREC LiveQA-Med (Ben Abacha et al, 2017): https://github.com/abachaa/LiveQA_MedicalTask_TREC2017
5) MEDIQA-2019 datasets on NLI, RQE, and QA (Ben Abacha et al., 2019): https://github.com/abachaa/MEDIQA2019 
6) MEDIQA-AnS dataset of question-driven summaries of answers (Savery et al., 2020): https://osf.io/fyg46/ Paper: https://www.nature.com/articles/s41597-020-00667-z 
7) MedQuaD Collection of 47k QA pairs (Ben Abacha and Demner-Fushman, 2019): https://github.com/abachaa/MedQuAD 
8) Medication QA Collection (Ben Abacha et al., 2019): https://github.com/abachaa/Medication_QA_MedInfo2019
9) Consumer Health Question Summarization (Ben Abacha and Demner-Fushman, 2019): https://github.com/abachaa/MeQSum
10) emrQA: QA on Electronic Medical Records (Pampari et al., 2018). Scripts to generate emrQA from i2b2 data: https://github.com/panushri25/emrQA 
11) EPIC-QA dataset on COVID-19 (Goodwin et al., 2020): https://bionlp.nlm.nih.gov/epic_qa/ 
12) BiQA Corpus (Lamurias et al., 2020): https://github.com/lasigeBioTM/BiQA  Paper:https://ieeexplore.ieee.org/document/9184044 
13) HealthQA Dataset (Zhu et al., 2019): https://github.com/mingzhu0527/HAR Paper: https://dmkd.cs.vt.edu/papers/WWW19.pdf 
14) MASH-QA Dataset on Multiple Answer Spans Healthcare Question Answering, with 35k QA pairs (Zhu et al., 2020): https://github.com/mingzhu0527/MASHQA Paper: https://www.aclweb.org/anthology/2020.findings-emnlp.342.pdf 

II) Medical VQA Datasets (Radiology): 
=====================================

1) VQA-RAD (Lau et al. 2018): https://osf.io/89kps
2) VQA-Med 2018 (Hasan et al. 2018): https://www.aicrowd.com/challenges/imageclef-2018-vqa-med
3) VQA-Med 2019 (Ben Abacha et al. 2019): https://github.com/abachaa/VQA-Med-2019
4) VQA-Med 2020 (Ben Abacha et al. 2020): https://github.com/abachaa/VQA-Med-2020 


III) Online QA Systems:
========================
-- I searched and tested several systems (e.g. AskHERMES, MiPACQ, SimQ). This list includes only the systems that are still maintained.

1) CHiQA (Consumer Health Question Answering System): chiqa.nlm.nih.gov
2) Neural Covidex: covidex.ai


IV) Medical Datasets Relevant to Question Answering: 
=====================================================

1) i2b2 shared tasks (2006-2016): www.i2b2.org/NLP
2) n2c2 NLP clinical challenges (2018-2019): https://n2c2.dbmi.hms.harvard.edu
 https://dbmi.hms.harvard.edu/programs/national-nlp-clinical-challenges-n2c2
3) TREC	Medical	Records Track (2012-2013).  
4) TREC	Clinical Decision Support Track (2014-2016): http://www.trec-cds.org	
5) TREC	Precision Medicine Track (2017-2019): http://www.trec-cds.org
6) CLEF	eHealth (2013-2020): https://clefehealth.imag.fr 
7) COVID dataset (CORD-19): https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge 




