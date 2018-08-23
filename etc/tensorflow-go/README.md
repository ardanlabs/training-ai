## Dockerfile for gophernotes

install tensorflow-go image
```
$ cd gophernotesDocker

$ docker build . -t gophernotesLocal/tensorflow-go
```

Run the built image

```
docker run -it -p 8887:8888 -v PATH-TO-YOUR-PROJECT/training-ai/machine-learning-with-go/ml_with_go:/go/src/machine-learning-with-go dicaormu/tensorflow-go
``` 

run bash in the image to compile your project

```
docker exec -it IMAGE_ID /bin/bash
```
