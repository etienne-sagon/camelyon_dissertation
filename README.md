# Deep Learning for Breast Cancer Detection
## MSc Data Intensive Analysis Dissertation Project

### Abstract
The presence of metastases in lymph nodes is a major determinant of breast cancer diagnosis and treatment. The detection of metastases is usually done manually by an expert pathologist. Computer-assisted diagnosis methods have been developed to automate the detection process and speed up diagnosis. However, most of these methods are computationally intensive and incompatible with a clinical application.  In this paper, a comprehensive pipeline is built to classify patches extracted from Whole Slide Images. The method proposed consists of an extensive preprocessing part, and the use of a pretrained GoogLeNet. The results are insufficient for clinical practice but perspectives for future improvements are given.

Content of the project Directory:

camelyon16/ :

  - src/ : 
      - preprocessing/ : includes scripts for data augmentation, patch extraction, and operations on WSIs
      - training_models/ : includes scripts to load and define our custom GoogLeNet and train it
      - inference/ : includes scripts for evaluating the model on the validation and test set
        
  - main.py : pyhton script executed during jobs
    
  - build_docker_image.sh : sh script to build the Docker image
    
  - run_job.sh : sh script to run a job

  - logs/ : includes logs of preprocessing and training

  - metrics_test/ : includes the logs from the evaluation on validation and test set

  - models/ : includes the weights of the GoogLeNet trained
