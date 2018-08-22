FROM alpine:3.7

# Jovyan user
ENV NB_USER jovyan
ENV NB_UID 1000
RUN adduser -s /bin/bash -u $NB_UID -D $NB_USER
USER jovyan
RUN mkdir /home/$NB_USER/work && \
    mkdir /home/$NB_USER/.jupyter && \
    mkdir /home/$NB_USER/.local

USER root

# Install Jupyter and gophernotes.
RUN set -x \
    # install python and dependencies
    && apk update \
    && apk --no-cache add \
        ca-certificates \
        python3 \
        su-exec \
        gcc \
        g++ \
        git \
        pkgconfig \
        python3-dev \
        zeromq-dev \
        musl-dev \
        mercurial \
        libtool \
        autoconf \
        automake \
        make \
    && echo pip upgrade && pip3 install --upgrade pip \
    && echo python3.6 && cp /usr/bin/python3.6 /usr/bin/python \
    ## install Go
    && echo update chacher && apk --update-cache --allow-untrusted \
        --repository http://dl-4.alpinelinux.org/alpine/edge/community \
        --arch=x86_64 add \
        go \
    ## jupyter notebook
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && pip3 install jupyter notebook \
    ## clean
    && find /usr/lib/python3.6 -name __pycache__ | xargs rm -r \
    && rm -rf \
        /root/.[acpw]* \
        ipaexg00301* \
    && rm -rf /var/cache/apk/*

# switch to jovyan user
USER jovyan

# environment variables
ENV GOPATH /home/jovyan/go
ENV CGO_ENABLED=1 CGO_CPPFLAGS="-I/usr/include"

# install gophernotes
RUN set -x \
    && GOPATH=~/go \
    && go get -insecure github.com/pebbe/zmq4 \
    && go get github.com/gopherdata/gophernotes \
    && mkdir -p ~/.local/share/jupyter/kernels/gophernotes \
    && cp -r ~/go/src/github.com/gopherdata/gophernotes/kernel/* ~/.local/share/jupyter/kernels/gophernotes

# move the gophernotes binary
USER root
RUN cp /home/jovyan/go/bin/gophernotes /usr/local/bin/
USER jovyan

# get the relevant Go packages
RUN set -x \
    && GOPATH=~/go \
    && go get -insecure gonum.org/v1/plot/... \
    && go get -insecure gonum.org/v1/gonum/... \
    && go get github.com/kniren/gota/... \
    && go get github.com/sajari/regression \
    && go get github.com/sjwhitworth/golearn/... \
    && go get -insecure go-hep.org/x/hep/csvutil/... \
    && go get -insecure go-hep.org/x/hep/fit \
    && go get -insecure go-hep.org/x/hep/hbook \
    && go get github.com/montanaflynn/stats \
    && go get github.com/boltdb/bolt \
    && go get github.com/patrickmn/go-cache \
    && go get github.com/chewxy/math32 \
    && go get gonum.org/v1/gonum/mat \
    && go get github.com/chewxy/hm \
    && go get -u gorgonia.org/vecf64 \
    && go get -u gorgonia.org/vecf32 \
    && go get github.com/awalterschulze/gographviz \
    && go get github.com/leesper/go_rng \
    && go get github.com/pkg/errors \
    && go get github.com/stretchr/testify/assert \
    && go get github.com/kniren/gota/dataframe \
    && go get github.com/skelterjohn/go.matrix\
    && go get github.com/gonum/matrix/mat64\
    && go get github.com/gonum/stat \
    && go get github.com/mash/gokmeans \
    && go get github.com/garyburd/go-oauth/oauth \
    && go get github.com/machinebox/sdk-go/textbox

EXPOSE 8888
CMD [ "jupyter", "notebook", "--no-browser", "--ip=*",  "--NotebookApp.token=''", "--NotebookApp.disable_check_xsrf=True" ]
