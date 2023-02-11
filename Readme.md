# Emotion recognition
### Introduction 
The following project is an API and development environment to detect emotions based on *.wav* files. Model was trained on images that are *MEL* spectrograms of audio files. Model is trained from scratch and uses AlexaNet architecture to classify emotions.


![image info](./documentation/dataset_summary.png)


### Build 
*run_docker.sh* script contains all necessary command to build and run container.
It will run container and expose API and JupyterNotebook server on ports *4444* and *8888*.

 ```sh
 chmod +x run_docker.sh
 docker.sh
 ```

### Accessing JupyterNotebooks 
![image info](./documentation/jupyter.png)
### Querying Flask API 
Api was tested using postman. It can be queried using the following configuration.
![how to run a training](./documentation/api_train.png)
![how to query a prediction](./documentation/api_predict.png)