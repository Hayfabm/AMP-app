
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


# Project Title

A brief description of what this project does and who it's for



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

![Logo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAw0AAADdCAYAAAD5N55SAAAIOnRFWHRteGZpbGUAJTNDbXhmaWxlJTIwaG9zdCUzRCUyMmFwcC5kaWFncmFtcy5uZXQlMjIlMjBtb2RpZmllZCUzRCUyMjIwMjEtMTAtMjlUMDklM0EwNyUzQTM4LjMxM1olMjIlMjBhZ2VudCUzRCUyMjUuMCUyMChYMTEpJTIyJTIwZXRhZyUzRCUyMkFVUFQ2YmF6bjl3eGxUZ0V2bDQyJTIyJTIwdmVyc2lvbiUzRCUyMjE1LjUuNiUyMiUyMHR5cGUlM0QlMjJkZXZpY2UlMjIlM0UlM0NkaWFncmFtJTIwaWQlM0QlMjJLejlzQjRFYzlfcnJST3pac3JfVCUyMiUyMG5hbWUlM0QlMjJQYWdlLTElMjIlM0U3VnBiZDZJOEZQMDFQdlpiWUFEeDBYcVpkdG5wbXRaWjAlMkJtOFJZaVlLUklueHFyOTlWOGlBWVdrVmp0YzZ1cjRJamtrSWV4OTlqbTUwQURkMmZvTGhmUHBWJTJCS2pzTkUwJTJGSFVEOUJyTnBna2NoJTJGOEp5eWEydE5vZ05nUVUlMkI3TFN6akRDTDBnYURXbGRZaDh0TWhVWklTSEQ4NnpSSTFHRVBKYXhRVXJKS2x0dFFzTHNVJTJCY3dRSXBoNU1GUXRUNWduMDFqcTJzYk8lMkZzVndzRTBlYkpweURzem1GU1doc1VVJTJCbVMxWndMOUJ1aFNRbGg4TlZ0M1VTakFTM0NKMncxZXVac09qS0tJSGRQZzE4djFlTEQ2OWJoNWVoclA3NGZCajhHUDVZVWoyWGlHNFZLJTJCc1J3dDJ5UVFVTEtNZkNSNk1ScmdjalhGREkzbTBCTjNWNXgwYnB1eVdjaExKciUyQmM0REFjeWJiUGlQS1JYWW8lMkZ6Qkh0aERpSXVKa1IwVVlkdm53alVSMnQ5MHp5ZGI0Z01rT01ibmdWZVRjaFFicFdXbDd0aUdxNTBqYmRJeW1sQkVybkNOS3VkJTJGanhDd25oQ1hCYUxRVTk1SE4lMkZra1ZDMlpRRUpJSmhmMmU5ek9LN3EzTkRCRkJiVkg4anhqWlNISERKU0JiekJhUGtDWFZKU09qMm1jRFklMkZnNkJ2Q0JMNnFFRGJ5TGRRSXolMkJJQlVVaFpEaDU2eGFkTGpLcHQ4STVrTkpLV3dhV1E2QlkyZTdZSkFHaU1sV09YYlNZYnlmc0tiRyUyRloxUWVLMlBuJTJGbGx3TFlveHFZSjJRN2RTNEYyJTJGaXhKWElGRDdqamlYWGFtdUcybjN4dU83dnU5dTZRVFBzeTRuMnpmM0t4NVlrR0R1TG0lMkZlaGclMkJQbHpWT1liUiUyRmQzVjk4N3c3cjdPUVhTdmU5ZWQzdU93UHhxY09JeWNxdCUyQklnbkF4ajVQUkJLJTJCRnNvdUlkcTJjVXBwcXRHdHJnbDI3dEZoWFclMkJydzRXSzY3VFJwdFJmNmZCdTV2cVVMaW01ekxHWWpoWEFCUUk0TVV5VURPQm8yVExNME9xdzZVZzlhWSUyRlp6NyUyRnBSZFBXZkxVdTl0ZXg1VzlqSVFobnBDclNyeVZldFBQRVY1eXV6ZlM0c3Y1JTJGTFJDSmxjd25jSEpmdWNWeDJLSVdidldwelVXRng0RG0yM21jR3g5WjM3WndyeFNNbzFyR01EeEhOQzRqTnlwUlNFNXZOWnFXeDJYVHFVTzJyV0pZdUxNV0JjOERHSVVBUjF0c0t0U3BTcUZXQjR0b2F3U21UWFRRYkk5JTJGSFVkQVFIaHZmSHRPVDU4d2gzQ0I2ZlB0S2RBOWx5ZVB1eVFkWGpQYk5WeHhtWCUyRnVHUnZ0V1dkSUhhaGo5b0FtN2pHbVpXZEUyZ3Exc0klMkJUNExDaFFXUGxGbUhVNGxTdjFIYU5SZW1CSlFOJTJCTExOM2IyNHBVWFlDSWxhZ1AxQVN1VzF1QjBqUU1QcldHUVUwYWJsV2tZZmRFRGJlcTBMQzZ1WEl6JTJCdjcxakVYY3FsbkU5dWZlSDdFcUVyR1ZUM2pWN284azc3bW5HNDlFSG1Ubm94ekx5VUdvT2RheUswMSUyRjZrcGxnamh5NGpTVnJpRDFHJTJCa1I1eG5BcTNpb0J0NTBKNkFTZkMxSHdmZkFzY2M0UFZtS0dKNWhqNUl4aHVGYnk4UHh1NDlaMHBiZktQS3h4ekNKJTJGdnBoT1YlMkZoekxHc1F5aHJ3N3c3ekxEdnh4RWFMZkFMSEclMkI3RXA0bXN6JTJGdjE3NXMyRDNSRnclMkZLaXpnJTJCRjdURHBHd0NhMXhJZHhSVFhtNnpGUSUyRnF6SEJFdUtuallh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MWM4evToY7craM9famoqbdu2pWnTprl2zKxyubZs2UL58uW1X30tgeDgYINhoLmNpUtDfh+elD4vIX1Ur64rTJMdaXBzc9OO97TSkJaWRtGiRQ16GsaOHctnn30G6J6Hzp07a6/Xvn17lFLa8KgVK1aglOKjjz7SXtPX11fbvm7dutjZ2ZGSkmIgA82bN8fLy4vr169z8eJFkQaAObvmaL/gZ1caeq7sid0AO7448IW2bMaOGdgNsGPk+pHasowlVw9dOIRTkBMnrmfOws9KGkAnKXYD7LgUfUlb1jqiNXYD7Nh2VPdBWVikwVzDk06cOEGlSpUMlv3xxx/cu3cvR9Lw4osvastTUlJwd3fnwoULvPnmm3z55Zfautu3bxMVFcWGDRsMvnjv3LmDjY0NUVFRBAUFGVyPb7/9NktpuHHjhradXhqmT5/O4MGDteWffvqp2aUhu9vJ8KT8SWGThpYtW7Jv377HblfQnr8///yT//znP7l6zKyk4fXXX2fDhg3ZPk5CQkJunpbR4xV2acjPidDJycm4u7tTo0YNg+V9+vTRXffjxx8rDW3atMHGxsZobsKTSAPAO++8g7W1NaGhofTt2xelFIsX68rvZ5SGr7/+GqUUrVq1YvHixZQrVw4nJyeuXr1qkNPQt29fgoODsbKy0n6cTC8D1apVw9HRkYiICFq0aIFSijfffDPTdk+L2aWhyIAiOAx00OK9Ve8BxqXhm7++wTPEk3sP72VbGlb+shIVqLh656q27MjFI6hAxdajW7VlGaUBYNymcTSY3sCgihKYloaeK3tSeUJlg2Vzds3Bpp8Ndx/cBQqPNOR24zO7pKamUrNmTZYuXUpKSgpbt26lZMmS3L9/P0fSoJTim2++ITU1lenTp2sSsW7dOl544QWuXbtGTEwMrVq1YvLkycTGxuLi4sKWLVtITU0lLCwMKysroqKi2LdvH2XKlOHUqVM8fPiQdu3aUapUKSB70vD3339TpkwZzp49y927d2nQoEG+kYaClAhd2OJZNnojIyN58cUXcXFxoXXr1toY35iYGN58803c3d2pVKmSJuxRUVE0atSIt956Cw8PD/z9/fn+++8BneSPGTOGsmXLUr58eRYs0BXQePjwIX379qV06dKUKFGCTp068eDBA6ZMmYKjoyPly5dn//79j33uzfX85TYPHjygQoUKODg4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)

## Specifications

This work is based on the paper : Deep learning improves antimicrobial peptide recognition.  Models and datasets are made freely available through the Antimicrobial Peptide Scanner vr.2 web server at www.ampscanner.com.


## Specifications

This work is based on the paper : Deep learning improves antimicrobial peptide recognition.  Models and datasets are made freely available through the Antimicrobial Peptide Scanner vr.2 web server at www.ampscanner.com.
