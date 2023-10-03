FROM python:3.10.12-bookworm
# Pre-installed library and its version

WORKDIR /root/code

RUN pip3 install --upgrade pip
RUN pip3 install ipykernel
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install mlflow
RUN pip3 install seaborn
RUN pip3 install ppscore
RUN pip3 install dash
RUN pip3 install shap
RUN pip3 install dash_bootstrap_components

RUN pip3 install dash[testing]
RUN pip3 install pytest
RUN pip3 install pytest-depends

# RUN pip install mlflow==2.6.0 cffi==1.15.1 defusedxml==0.7.1 matplotlib==3.7.2

COPY ./code /root/code  

CMD tail -f /dev/null
