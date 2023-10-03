# Machine Learning Assignment3 : Car Price Prediction (Logistic Regression)
### Name: Sitthiwat Damrongpreechar
### Student ID: st123994

### Deployed website URL: https://st123994_a3.ml2023.cs.ait.ac.th/

#### Required applications:
1. Visual Studio Code (VScode)
2. Docker Desktop
   
#### Required VScode extensions:
1. JupyterNoteBook
2. Python
3. Docker
4. Remote Development packages (optional)
5. Dev Containers (optional)

#### How to use:
1. Download or gitclone this repository.
2. Open your Docker Desktop for building images and composing.
3. Get into the folder 'app' and check the Docker files (.Dockerfile for python, mlflow.Dockerfile for Mlflow).
4. After building, to operate only local website, compose up the 'docker-compose.yaml' .
   -  To operated the 'assignment3.ipynb' and local website, compose up the file 'docker-compose_arch.yaml' (move file into './app/code' first). 
   - 'docker-compose-deploy.yaml' is the outline for deploying a docker compose in ml2023 server.
5. Drive into remoted docker container name:
   - 'sittiwat555/assignment3' for 'docker-compose.yaml', select 'open in browser' to open the website
   - 'assignment3' for 'docker-compose_arch.yaml', select 'attach in vscode'
   - 'assignment3-mlflow' for 'docker-compose_arch.yaml', select 'open in browser' to open the mlflow ml2023
6. For the 'docker-compose_arch.yaml, execute the 'main.py' file in folder 'code' to run the local website.
7. Install extensions that VScode suggests.
8. Enjoy your prediction and Website!  
