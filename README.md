# 

## Libraries and Frameworks

- PyTorch: The foundation for building custom deep learning models, including CNN and biLSTM architectures. Enables flexible model design and efficient computation, with support for GPU acceleration. Facilitates the integration of Data objects, optimal for developing within this framework.

- SageMaker: Offers a managed environment for training, tuning, and deploying models at scale. Supports Hyperparameter Optimization (HPO) and provides access to powerful AWS instances for accelerated model training. Finally, but not less important, provides seamless deployment options, including real-time and batch inference, allowing for scalable model serving.

- smdebug: Part of AWS SageMaker's tools, it assists in tracking and debugging models during training by logging and visualizing essential metrics, such as gradients and losses. Facilitates model overseeing through automated rule-based monitoring to detect common issues like vanishing gradients and overtraining, improving model stability. Reduces manual inspection efforts, allowing for a streamlined debugging process in deep learning workflows.

## Content and Folder descriptions

- **root dir**
Several files are located in the root dir as this structure is easier for model trainig and deploying - it's also the common structure to further develop and API or web application. This director contains the notebook for the project development process, python scripts required to train and deploy the model (training job submission and entry points for training and inference).As well, the capstone proposal and blog post are stored under this path in PDF format. images for proving the succesful execution of the state machine and its definition in json format.

**data**: Contains all data related to this project: original dataset, predictions and train/validation/test splits.

**images**: Visualization for NN architectures and data exploration.

**models**: model classes, and trained model in pth format.

*scripts*: Code to train the benchmark model since it was not run from the development notebook.

**utils**: Auxiliary code pertaining data loading and processing, incluiding our One Hot Encoder class.

## Dataset citation

Deep learning regression model for antimicrobial peptide design
Jacob Witten, Zack Witten
bioRxiv 692681; doi: https://doi.org/10.1101/692681