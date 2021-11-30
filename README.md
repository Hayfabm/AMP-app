# AMP-app
deepchain.bio | Antimicrobial peptide recognition 

## Install AMP conda environment

From the root of this repo, ```run conda env create -f environment.yaml```

Make sure you've tensorflow==2.5.0 installed

Follow this [tutorial](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras#step-5-monitor-your-tensorflow-keras-training-in-neptune) to make neptune logger works

## Abstarct: 

Antimicrobial peptides (AMPs) are a class of small peptides that widely exist in nature and they are an important part of the innate immune system of different organisms. AMPs have a wide range of inhibitory effects against bacteria, fungi, parasites, and viruses. The emergence of antibiotic-resistant microorganisms and the increasing concerns about the use of antibiotics resulted in the development of AMPs, which have good application prospects in medicine, food, animal husbandry, agriculture, and aquaculture. Faced with this reality, Machine learning methods are now commonly adopted by wet-laboratory researchers to screen for promising candidates in less time. In this app, we propose a deep convolutional neural network model associated with a long short-term memory layer called CNN-LSTM, for extracting and combining discriminative features from different information sources in an interactive way. 
Performance Details: using the 10-Fold Cross-Validation, our model outperforms 90.0% (Standard Deviation: +/-1.69%) accuracy, 87.28% (+/-2.69%) sensitivity, 93.32% (+/-2.38%) specificity, 80.81 (+/-3.36) MCC,  96.02% (+/-0.99%) roc_auc and 96.63% (+/-0.73%) roc_pr.



## Model Architecture:

![Model architecture](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=Untitled%20Diagram.drawio.png#R7VpZc9o8FP01POYbG9kGHglLk0maaUKnafombAFqjOUKESC%2F%2FpOwvEoBknpJJuUF61qb7znnam2BwXL7hcJw8ZV4yG%2B1DW%2FbAsNWu20Cx%2BF%2FwrKLLB2rFxnmFHsyU2qY4GckjYa0rrGHVrmMjBCf4TBvdEkQIJflbJBSsslnmxE%2F32oI50gxTFzoq9Z77LFFZO3aRmq%2FQHi%2BiFs2DflmCePM0rBaQI9sMiYwaoEBJYRFT8vtAPnCebFfonLjF94mHaMoYKcU%2BPV8OR1vfj3sHh%2Bn4d3V%2FMf4x%2FrMAVE1T9Bfyy%2BWvWW72AWUrAMPiVqMFjjfLDBDkxC64u2Gg85tC7b0ecrkjzPs%2BxNZ9glR3rNz8Ye5R%2Fs%2BngfczIgoo3ZffpHIjrYZk%2FycL4gsEaM7nkW%2BdaRnJbXath2lNylQna7Ms8iAlEACJTnmSdWp%2F%2FiDdOEr3Gl1FO8hj%2FNJJgllCzInAfRHqfU87980zzURjtp79TdibCfFAdeM5H2%2BYpQ8ogHxCd23CYz975CTV2RNXXTgSyQNRO8PQkGRDxl%2ByqtF51dZ9BvBvCsJhG0jjyFw7HwVDNI5YrJUAZ2kG28HrK2hv%2BML1nr4iT%2FO2d6LkWlG9l13E0c7f9YkysBd7jjiW1JTVLY%2FGl5N7kbD27gS3s2onnzd3KxpsaROXN9d3F893F802YfJ3e3F9%2F7V7V2TnRhcDi%2F7w4er0WT8ym4UVH0kCsJVGA1GM7wVyi4j2nUKSomDWCba9TTBrldZrGts6PDgarGvNC6VCX2ejbqepQuK3fZUzEZKwQKAAhimCgZwNGiYZmVwWE0MPWiL2c%2FM84Oo6j9bpoZbWfM%2BsZOJKoYr0KtnvOoUga95vDJ7HwXlt2MZS6RqLEG3gGX3NCz7lMJdJlsoMqwOtGPrOTM%2BNX%2FXLlAp6kG5xDLeRTQvITYrU0pNbDbbtcZm02lCtS%2F6snJhKQQuODYKAYqwjivUqkmhVg2K62kEp0x20XKKPA8H85ZgbPR6Sl89Z%2FbhDtHTy9eieyhTLqcn71w52jdfIExW%2B4ZG%2B1ZV0gdqGH2nA3YV0zKzpm0EW9lGKOBZUqCwiosw6%2FBQruR3jFblgSV2eiayDG5ualJ1CSJWoj5QB3Dd2gpUpmHwqTUMGtJwpyYNd1%2Bp4U4dGlY3V64n379%2BYBF3Ghax%2Fbn3R6yaRGwVB7x690fi78zoxiWBC9nHUY5VONZKFicZ5di1Dn%2FqSmWGuOfEaSrdQOq1kiPOD%2BBehaEa9yY7AbX413IU%2Fx449pgmJ0sBw0vsUjLF0M%2Bs5KZvPlFJSoYoZNiboYMrxhLa%2BUaRh12GSXBseXu0sQL9OBlYnmPKcrPIsCX2vCjooxV%2BhtN9VYK8ckLB67XPW%2FZQ1MXj%2FCoK%2BSVtWin7ypbKSt3pjl3ZcGkrpOwvcUC4qe9iIfkV%2BrNGAVd00fdliT0%2FDgYkEOCIs7q4tOYKxN%2BeqrU10xTtZmFlVwhsNRpMEk8b%2FI9E20FNEl6PTOYgriKwzJ6dR6urQQvUilZHQWt4c3Omnat%2FCoiKqyUdRNo7OZUh1H1xdJU3CFKI4usC4sVZ5Mw%2Bz2Ca4Va9S8D9YOT2VZWbBB8F%2F9xSY39xoqQlYF6tlnHaRLYyKsQT639UaJoKujO5eqlg%2FqNCM1Swle23pqnQPkaFV10xw0G4Zu8S%2FDImYMW9Cc102SoHPJ5M70ZH20PpDXMw%2Bh8%3D)


## Specifications

This work is based on the paper : Deep learning improves antimicrobial peptide recognition.  Models and datasets are made freely available through the Antimicrobial Peptide Scanner vr.2 web server at www.ampscanner.com.

