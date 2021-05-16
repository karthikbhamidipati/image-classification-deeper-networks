# image-classification-deeper-networks
Image classification using deeper networks like ResNet, VGG, GoogLeNet etc.,

# Requirements
1. Run `pip install -r requirements.txt` from the project directory to install the requirements.
2. Data will be stored in `wandb`, so it should be installed and account has to be created. Also get the api key from wandb account online and add `WANDB_API_KEY` to the environment variables.

# Run instructions
1. To `train` the model, run
    ```shell
    python main.py train -r <data_path> -d <dataset_name> -m <model_name> -s <model_save_path> 
    ```
   **Models Implemented:** resnet18, resnet34, resnet50, resnet101, resnet152, vgg11, vgg13, vgg16, vgg19, GoogLeNet  
   **Datasets Included:** mnist, cifar10, cifar100  
   **Note:** Download MNIST dataset manually and place it in the data_path as it's website is closed down.

1. To `test` the model on the test data, run
    ```shell
    python main.py test -r <data_path> -d <dataset_name> -m <model_name> -s <model_save_path> 
    ```
   **Models Implemented:** resnet18, resnet34, resnet50, resnet101, resnet152, vgg11, vgg13, vgg16, vgg19, GoogLeNet  
   **Datasets Included:** mnist, cifar10, cifar100
   