
# AMP-app
deepchain.bio | Antimicrobial peptide recognition 🦠

## Install AMP conda environment

From the root of this repo, ```run conda env create -f environment.yaml```

Make sure you've tensorflow==2.5.0 installed

Follow this [tutorial](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras#step-5-monitor-your-tensorflow-keras-training-in-neptune) to make neptune logger works

## Abstract: 
Antimicrobial peptides (__AMPs__) are a class of small peptides that widely exist in nature and they are an important part of the innate immune system of different organisms. AMPs have a wide range of inhibitory effects against bacteria, fungi, parasites, and viruses. The emergence of antibiotic-resistant microorganisms and the increasing concerns about the use of antibiotics resulted in the development of AMPs, which have good application prospects in medicine, food, animal husbandry, agriculture, and aquaculture. Faced with this reality, Machine learning methods are now commonly adopted by wet-laboratory researchers to screen for promising candidates in less time. In this app, we propose a deep convolutional neural network model associated with a long short-term memory layer called CNN-LSTM, for extracting and combining discriminative features from different information sources in an interactive way. 

__Performance Details__: using the 10-Fold Cross-Validation, our model outperforms 90.30% (Standard Deviation: +/-1.69%) Accuracy, 87.28% (+/-2.69%) Sensitivity, 93.32% (+/-2.38%) Specificity, 80.81 (+/-3.36) MCC,  96.02% (+/-0.99%) ROC_auc and 96.63% (+/-0.73%) ROC_pr.


## Model architecture:
![Logo](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/34/16/10.1093_bioinformatics_bty179/1/m_bty179f1.jpeg?Expires=1641329434&Signature=iHc-Jf-UIHFw-oPFJV84Zcyrn9uPeTGi0jHnjhM7gdLin9FgFcr~Wmr4t-S5XpucznqlPadO5t0KRDOstGXiOmQMzUHmtQL9bFIwaqHGLER4zl216rrLx4xWYslx7p5mBGTx~wPNsLkcoc~BunJc8HGlcKkgHUXUBNNdc6E~NvOmGVBo9awlhN4ARSV0-qr-exorjQNYJxQ3xGkpenEEngw6rv71UIOk0lPdLUJFeiiwfGj6DCa~mougw0u4opSF9CXpVmTERDMM~fSC4Be7ciSWQWlrctnklnqXBJrKzqjenL5vQJOd2htJtUZoH92XMXrUcJEYkdQzSGcdd6wfTQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

## Specifications:
This work is based on the paper : [Deep learning improves antimicrobial peptide recognition](https://academic.oup.com/bioinformatics/article/34/16/2740/4953367).  Models and datasets are made freely available through the Antimicrobial Peptide Scanner vr.2 web server at www.ampscanner.com.


