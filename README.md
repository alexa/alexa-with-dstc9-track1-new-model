## DSTC9 Track 1 - Can I Be of Further Assistance? Using Unstructured Knowledge Access to Improve Task-oriented Conversational Modeling

This repository contains the source code of the latest model developed for [DSTC9 Track 1](https://dstc9.dstc.community/) by the Conversational Modeling Science Team of Amazon Alexa AI.

The task formulation and details can be referred to [this link](https://github.com/alexa/alexa-with-dstc9-track1-dataset). The model details can be referred to this paper: [Jin, Di, Seokhwan Kim, and Dilek Hakkani-Tur. "Can I be of further assistance? using unstructured knowledge access to improve task-oriented conversational modeling." DialDoc at ACL (2021)](https://aclanthology.org/2021.dialdoc-1.16/). If you use this code, please consider citing:

```
@inproceedings{jin-etal-2021-assistance,
    title = "Can {I} Be of Further Assistance? Using Unstructured Knowledge Access to Improve Task-oriented Conversational Modeling",
    author = "Jin, Di  and
      Kim, Seokhwan  and
      Hakkani-Tur, Dilek",
    booktitle = "Proceedings of the 1st Workshop on Document-grounded Dialogue and Conversational Question Answering (DialDoc 2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.dialdoc-1.16",
    doi = "10.18653/v1/2021.dialdoc-1.16",
    pages = "119--127",
    abstract = "Most prior work on task-oriented dialogue systems are restricted to limited coverage of domain APIs. However, users oftentimes have requests that are out of the scope of these APIs. This work focuses on responding to these beyond-API-coverage user turns by incorporating external, unstructured knowledge sources. Our approach works in a pipelined manner with knowledge-seeking turn detection, knowledge selection, and response generation in sequence. We introduce novel data augmentation methods for the first two steps and demonstrate that the use of information extracted from dialogue context improves the knowledge selection and end-to-end performances. Through experiments, we achieve state-of-the-art performance for both automatic and human evaluation metrics on the DSTC9 Track 1 benchmark dataset, validating the effectiveness of our contributions.",
}

```

## How to use it?

1. Install the required packages by running:
```
pip install -r requirements.txt
```

2. Download data used for training and evaluation and trained model parameters from [this link](). Decompress it to the root folder:

```
tar -xvzf DSTC9-Track1.tar.gz -C ./
```

3. Run the following command to obtain knowledge-seeking turn detection predictions and the results are saved in the folder "pred":
```
sh ./ktd.sh
```
 The data used for training the knowledge-seeking turn detection model are provided in the folder of "data_ktd", which adds those data augmentation dialogues (please refer to the paper for details).

4. Run the following command to obtain domain detection predictions. We have three labels for domain detection: train, taxi, and others (this label includes hotel, restaurant and attraction).
```
sh ./domain_classification.sh
```

If you would like to re-train this domain classifier, the data is stored in the folder "data_domain_detection_multiwoz_dstc_3way".

5. Run the following command to extract entities in the dialogue for each utterance given the knowledge base and then the results are saved in the file of pred/entities_detected.test.json.
```
python entity-extraction.py data_eval test pred/ktd.test.json
``` 

 Based on obtained domain detection and entities extraction results, we need to merge them into the final entity file:
```
python merge-domain-entities-detection.py data_eval test pred/ktd.test.json pred/preds-domain-detection.json pred/entities_detected.test.json
```

6. Run the following command to obtain knowledge selection predictions. To be noted, we have provided the best extracted entities for the test set in the file of "pred/entities_detected.test.best.json" together with our best knowledge-seeking turn detection predictions in the file of "pred/ktd.test.best.json" (this best result file is obtained by model ensembling). If you would like to test on our provided best results file, please accordingly revise the arguments of "entities_file" and "labels_file". 
```
sh ./ks.sh
```

7. Run the following command to obtain response generation predictions. After step, the final results file for official evaluation is "pred/pegasus-large-response-final.json" (such a result file has been provided for reference). To be noted, the trained generation model provided here is trained on truncated ground truth responses that have removed those transitioning questions.
```
sh ./generation.sh
``` 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License Summary

The documentation is made available under the Creative Commons Attribution-ShareAlike 4.0 International License. See the LICENSE file.

The sample code within this documentation is made available under the MIT-0 license. See the LICENSE-SAMPLECODE file.
