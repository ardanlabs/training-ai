# based in the dockerfile of https://github.com/ctava/tensorflow-go/blob/master/Dockerfile
FROM tensorflow/tensorflow

#Begin: install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git
#End: install dependencies

#Begin: install golang
ENV GOLANG_VERSION 1.10.3
ENV GOLANG_DOWNLOAD_URL https://golang.org/dl/go$GOLANG_VERSION.linux-amd64.tar.gz

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN curl -fsSL "$GOLANG_DOWNLOAD_URL" -o golang.tar.gz && \
    tar -C /usr/local -xzf golang.tar.gz && \
    rm golang.tar.gz && \
    mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"
WORKDIR "/go"
#End: install golang

#Begin: install tensorflow
ENV TF_TYPE "cpu"
ENV TARGET_DIRECTORY /usr/local
RUN  curl -L \
        "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.10.0.tar.gz" | tar -C $TARGET_DIRECTORY -xz
RUN  ldconfig
ENV LD_LIBRARY_PATH $TARGET_DIRECTORY/lib
ENV LIBRARY_PATH $TARGET_DIRECTORY/lib
RUN go get github.com/tensorflow/tensorflow/tensorflow/go