# AWS Machine Learning Engineer Nanodegree
## Capstone Project Proposal
Omar Daniel López Olmos
September 18 2024

## Domain Background

Antibiotics are crucial for treating infections and supporting immune-suppressing medical procedures. However, their overuse has led to multi-drug resistant bacteria, causing over 700,000 deaths annually (Llor, 2014; WHO, 2014). Antimicrobial peptides (AMPs), produced by organisms' innate immune systems, offer a promising alternative by targeting bacterial cell membranes, making drug resistance unlikely (Zasloff, 2002). AMPs also have advantages like broad-spectrum activity, thermal stability, and low-cost synthesis (Mahlapuu, 2016; Raguse, 2002).

Recent AI advancements, including machine learning (ML) and deep learning (DL), have accelerated AMP discovery. DL models, often using peptide sequence data similar to Natural Language Processing tasks, have successfully optimized the in-silico design of AMPs (Bepler, 2021). Various models like RNNs, Variational Autoencoders, and GANs have shown promise (Müller, 2018; Dean, 2020), though most models focus on binary classification, predicting wether or not a given AMP has antimicrobial activity or not (Gabere, 2017).

### Problem Statement
For ML-guided AMP design the prediction of continous values is of great importance. The minimum inhibitory concentration (MIC) is a commonly used metric to assess antibiotic activity; the lower the MIC, the lower the required concentration of a peptide to prevent bacterial growth. But at the moment, not many models are tailored at predicting the MIC of a given AMP, and the ones that do, are limited to analyze a specific AMP motif, rather than the entire peptide sequence (Gabere and Noble, 2017; Hilpert, et al., 2006). The lack of tools that perform regression to assist in clinical developments of AMPs limit the scientific advances that could combat not only common infections but also upcoming outbreaks.

This project has the objective of building  regression systems that will predict the minimum inhibitory concentration (MIC) of a given AMP sequence. More specifically, the model predicts the MIC value on a logarithmic scale (pMIC) of a peptide, against one of the most prevalent bacterial pathogens in humans: *Escherichia coli.*

### Datasets and Inputs

The GRAMPA dataset is a publicly available collection of 6,760 unique AMP sequences and 51,345 MIC values from various peptide databases, including APD (Wang et al., 2015), DADP (Novkovic et al., 2012), DBAASP (Pirtskhalava et al., 2015), DRAMP (Kang et al., 2019), and YADAMP (Piotto et al., 2012). This study utilizes data for *E. coli* from GRAMPA. As outlined in Aldas Bulos et al. (2023), the dataset was filtered to exclude sequences with non-canonical amino acids and retain those with a length of less than 50 residues resulting in 4,567 sequences. I have already experimente with this dataset and after removing duplicate sequences, 4,540 unique sequences were retained. This dataset is in tabular format and contains peptide sequences in upper case letters each representing a single aminoacid, additionally, it contains experimentally measured properties for each sequence, however, in this project we will only be focusing on the sequence itself which will be properly encoded to allow DL models to ingest and interpret this type of data, in the end each peptide sequence will be encoded using the one-hot encoding schema. An interesnting paper from 2022 (Otović et al., 2022) hihglights that although DL models can benefit from both aminoacid physicochemical properties and sequence, the latter is more informative when it comes to predicting viral and antimicrobial activity, putting this into account it is acceptable to just analyze the sequential information contained in this dataset. 


### Solution Statement
We will use the encoded sequences as input data to train an ensemble model. This ensemble model is built from scratch and specifically designed to handle the complexities of amino acid sequence patterns. Our ensemble model will be composed of CNN layers followed by bidirectional LSTM layers, this architecture is intended to allow the model to extract both local and global dependencies present in the aminoacid sequence. Then, the processed output will be feed into a stack of linear layers to perform regression on our data.

### Benchmark Model
As a benchmark model we opted to build a vanilla CNN model with 6 convolutional layers. This architecture process the inputs through three stacked layers of two 1-D convolutional layers, which are then followed by three linear layers to perform regression. To regularize the model, a dropout layer is inserted before the information is passed to the regression module. 

### Evaluation Metrics

To evaluate the performance of our predictive models, we employed metrics commonly used in regression tasks, namely, mean squared error (MSE) (Eq. 1), root mean squared error (RMSE) (Eq. 2), mean absolute error (MAE) (Eq. 3), and the coefficient of determination ($R^2$) (Eq. 4): 

#### Mean Squared Error (MSE)
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

#### Root Mean Squared Error (RMSE)
$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

#### Mean Absolute Error (MAE)
$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

#### Coefficient of Determination (R²)
$$
R^2 = 1 - \frac{SS_{\text{residual}}}{SS_{\text{total}}}
$$

where $n$ is the total number of predictions, $y_i$ is the actual experimental pMIC value, and $\hat{y}_i$ represents the predicted value for for prediction $i$. $SS_{\text{residual}}$ is the sum of squared residuals, which measures the variance of the errors between the actual values ($y_i$) and the predicted values ($\hat{y}_i$). $SS_{\text{total}}$ is the total sum of squares, which measures the variance of the actual values from their mean. MSE measures the average of the squared differences between predicted values ($\hat{y}$) and actual values ($y$). A lower MSE indicates that the model's predictions are closer to the actual experimental pMIC values. RMSE is the square root of the mean of the squared differences between predicted values ($\hat{y}$) and actual values ($y$) and provides a measure of the average magnitude of error between our predictions and experimental values. MAE measures the average of the absolute differences between predicted values ($\hat{y}$) and actual values ($y$), and like the MSE and RMSE metrics, a lower value indicates that the model's predictions are closer to the actual values, signifying better accuracy.
Finally, $R^2$ represents the proportion of the variance in the dependent variable ($y$) that is predictable from the independent variable(s) ($\hat{y}$). A higher $R^2$ value is preferred since it indicates a better fit of the model to the data. 

### Project Design
The main issue addressed in our study is the absence of effective AMP activity prediction systems. Existing approaches predominantly function as binary classification systems or are limited to specific AMP motifs (Gabere and Noble, 2017. Hilpert et al., 2006). We will develop a deep learning ensemble model to predict the minimum inhibitory concentration (MIC) of AMPs against *Escherichia coli*. 

We propose a hybrid architecture where CNN layers act as the front-end to capture local sequence patterns in the one-hot encoded peptide sequences. The ability of CNNs to extract short-range features is key, as local interactions between amino acids are often crucial in determining antimicrobial properties. After passing through multiple convolutional layers, the sequence data is processed by biLSTM layers, which are designed to capture long-range dependencies in the data. This bidirectional mechanism allows the model to better understand both forward and backward interactions in the peptide chain, contributing to a more comprehensive sequence analysis. The processed output from the biLSTMs is then fed into a fully connected stack of linear layers, which performs regression on the pMIC values.

The input data to our model are peptide sequences recovered from various AMP databases and filtered to target *Escherichia coli*. As deep learning models require numerical inputs to perform computations, it is essential to convert the amino acid sequences into a suitable numerical format. In this project, we use one-hot encoding, a widely adopted method for representing categorical data, where each amino acid is transformed into a binary vector. This encoding allows the model to interpret the peptide sequences while retaining the positional and compositional information inherent to each sequence. 

Our model is intended to assist the clinial development of AMPs, so we decided to limit the peptide sequences to a maximum length of 50 residues, with that we optimize both model performance and real-world application. Shorter peptides are not only easier to synthesize but also more cost-effective, making them attractive candidates for therapeutic development. From a computational perspective, smaller peptides reduce the complexity of the model and the amount of data required for training, facilitating faster convergence and better generalization. Moreover, shorter sequences minimize the risk of overfitting, as the model can more efficiently learn relevant features without being overwhelmed by excessive sequence length. Finally, as a way of incorporating model regularization, and provide an extra way of avoiding overfitting, a dropout layer was added into the model.

-----------
#### References
 - Llor, C. and Bjerrum, L. (2014). Antimicrobial resistance: risk associated with antibiotic overuse and initiatives to reduce the problem. Therapeutic Advances in Drug Safety 5, 229–241. doi:10.1177/2042098614554919

- Organization, W. H. (2014). Antimicrobial resistance: Tackling a crisis for the health and wealth of nations. doi:\url{https://www.who.int/news/item/29-04-2019-new-report-calls-for-urgent-action-to-avert-antimicrobial-resistance-crisis}. Accessed:
545 2023-05-31
- Zasloff, M. (2002). Antimicrobial peptides of multicellular organisms. Nature 415, 389–395. doi:10.1038/415389a
- Mahlapuu, M., Hakansson, J., Ringstad, L., and Bj ˚ orn, C. (2016). Antimicrobial peptides: An emerging category of therapeutic agents. Frontiers in Cellular and Infection Microbiology 6. doi:10.3389/fcimb. 2016.00194
-  Gabere, M. N. and Noble, W. S. (2017). Empirical comparison of web-based antimicrobial peptide prediction tools. Bioinformatics 33, 1921–1929. doi:10.1093/bioinformatics/btx081
- Hilpert, K., Elliott, M. R., Volkmer-Engert, R., Henklein, P., Donini, O., Zhou, Q., et al. (2006). Sequence requirements and an optimization strategy for short antimicrobial peptides. Chemistry &amp Biology 13, 1101–1107. doi:10.1016/j.chembiol.2006.08.014
- Wang, G., Li, X., and Wang, Z. (2015). APD3: the antimicrobial peptide database as a tool for research and education. Nucleic Acids Research 44, D1087–D1093. doi:10.1093/nar/gkv1278
- Novkovic, M., Simuni ´ c, J., Bojovi ´ c, V., Tossi, A., and Jureti ´ c, D. (2012). DADP: the database of anuran defense peptides. Bioinformatics 28, 1406–1407. doi:10.1093/bioinformatics/bts141
- Pirtskhalava, M., Gabrielian, A., Cruz, P., Griggs, H. L., Squires, R. B., Hurt, D. E., et al. (2015). DBAASP v.2: an enhanced database of structure and antimicrobial/cytotoxic activity of natural and synthetic peptides. Nucleic Acids Research 44, D1104–D1112. doi:10.1093/nar/gkv1174
-  Kang, X., Dong, F., Shi, C., Liu, S., Sun, J., Chen, J., et al. (2019). DRAMP 2.0, an updated data repository of antimicrobial peptides. Scientific Data 6. doi:10.1038/s41597-019-0154-y
- Piotto, S. P., Sessa, L., Concilio, S., and Iannelli, P. (2012). YADAMP: yet another database of antimicrobial peptides. International Journal of Antimicrobial Agents 39, 346–351. doi:10.1016/j.ijantimicag.2011.12.00
-  Aldas-Bulos, V. D. and Plisson, F. (2023). Benchmarking protein structure predictors to assist machine learning-guided peptide discovery doi:10.26434/chemrxiv-2023-krc22[Preprint]