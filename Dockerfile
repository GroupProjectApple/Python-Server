FROM python:3.10.10-slim
WORKDIR /Collaborativefiltering
COPY . /Collaborativefiltering
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "Collaborativefiltering.py"]
