FROM python:3.6

# Install dependencies
RUN pip install -U \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    torch \
    torchvision

# Add our code
ADD infer.py /code/infer.py
