## Dockerfile for gophernotes

install jupyter for go OR build the docker image for jupyter with go 
in this folder
```
$ cd gophernotesDocker

$ docker build . -t gophernotesLocal/gophernotes
```

Run the built image

```
$ docker run -it -p 8888:8888 -v /PATH/TO/LOCAL/PROJECT/training-ai:/go/src/training-ai  gophernotesLocal/gophernotes

```