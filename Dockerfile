FROM python:3.10.10-slim
WORKDIR /Collaborativefiltering
COPY . /Collaborativefiltering
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "Collaborativefiltering.py"]
