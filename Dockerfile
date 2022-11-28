# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM jjanzic/docker-python3-opencv

RUN pip install --upgrade pip

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies.
#RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install streamlit==1.14.0 
RUN pip install tensorflow==2.10.0 opencv-python==4.5.5.64 roboflow

# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 9030
ENV TZ="America/Bogota"
RUN date

# Setting this ensures print statements and log messages
# promptly appear in Cloud Logging.
ENV PYTHONUNBUFFERED TRUE

# Copy local code to the container image.
COPY . .

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
#CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2 
CMD streamlit run main.py --server.port $PORT