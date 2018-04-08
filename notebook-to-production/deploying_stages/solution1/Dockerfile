FROM python

# Install dependencies
RUN pip install -U \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    torch \
    torchvision

# Add our code
ADD train.py /code/train.py
