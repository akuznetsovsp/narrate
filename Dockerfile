FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y git wget unzip gcc python3-dev espeak-ng git-lfs && \
    rm -rf /var/lib/apt/lists/*

RUN git-lfs install

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . $APP_HOME

RUN pip install --no-cache-dir --upgrade pip setuptools

RUN pip install -r requirements.txt

RUN pip install --no-cache-dir nltk && \
    python -m nltk.downloader punkt

RUN git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS $APP_HOME/ML_model/temp && \
    cd $APP_HOME/ML_model/temp && \
    git checkout 3aa7ba7f8f275ec13dce21682a61494c35089e2a && \
    git lfs pull && \
    cd $APP_HOME

RUN mkdir $APP_HOME/Data/reference_audio

RUN unzip $APP_HOME/ML_model/temp/reference_audio.zip -d $APP_HOME/Data && \
    mv $APP_HOME/ML_model/temp/Models/LibriTTS/epochs_2nd_00020.pth $APP_HOME/ML_model/Models/LibriTTS && \
    rm -rf $APP_HOME/ML_model/temp

EXPOSE 3000

CMD ["python", "app.py"]