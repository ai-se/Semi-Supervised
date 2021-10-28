# Semi-Supervised
[VanEngelen-Hoos2020_Article_ASurveyOnSemi-supervisedLearni.pdf](https://github.com/ai-se/Semi-Supervised/files/7339885/VanEngelen-Hoos2020_Article_ASurveyOnSemi-s!
upervisedLearni.pdf)

![alt text](https://user-images.githubusercontent.com/31140098/139264801-16387c05-693d-4254-8573-d025b8588ebc.png)

## Self-Training: (Widely used in SE)
1. Different supervised algorithms in semi-supervised setting

## Co-Training: (Widely used in SE)
1. Multi-view co-training: Different combinations of supervised algorithms
2. Single-view co-training: Different combinations of supervised algorithms
3. CO-Forest
4. Effort-Aware tri-Training

## Boosting:
1. SemiBoost

## Feature extraction:
1. Principal component analysis (not sure if we can call it semi-supervised)
2. FTcF.MDS

## Cluster-then-label:
1. Clsuter data using some clustering algorithm (EM) and then use the labels for assigning labels to clsuter then predict.
2.  semi supervised GMM

## Pre-training:
1. DNN - Not using it

## Maximum-margin methods:
1. S3VM
2. TSVM
3. S4VM

## Perturbation-based methods:
1. DNN - Not using it

## Manifolds:

## Generative Models:
1. DNN - Not using it

## Graph Based:
1. LabelPropagation
2. LabelSpreading




## Inductive methods:

### Wrapper methods (Multiple supervised methods in each)
1. **Self-training (Done)**
  1. **Co-training (Done)**
  1. **Multi-view co-training (Done)**
  2. **CO-Forest (Done)**
  3. **Single-view co-training (Done)**
  4. Co-regularization
1. Boosting
  1. SSMBoost 
  1. ASSEMBLE
  1. **SemiBoost (Done)**

### Unsupervised preprocessing
1. Feature extraction
2. **Cluster-then-label (Done)**
3. Pre-training

### Intrinsically semi-supervised methods
1. Maximum-margin methods
  1. **safe semi supervised Support vector machines(S4VM) (Done)**
  2. **Gaussian processes (working on now)**
  3. Density regularization
  4. **Pseudo-labelling as a form of margin maximization (working on now)**
2. Perturbation-based methods
  1. Pseudo-ensembles
  2. pi-model
  3. Temporal ensembling
  4. Mean teacher
  5. Semi-supervised mixup
### Others
  1. **LabelPropagation (Done)**
  2. **LabelSpreading (Done)**
  3. **Semi GMM (Done)**
  4. **EATT (Done)**
  5. **FTcF.MDS (Done)**
  6. **S3VM (Done)**
