FROM alpine:3.7

# Add gophernotes

ENV CGO_ENABLED=1 CGO_CPPFLAGS="-I/usr/include" GOOS=linux

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
        py3-zmq \
        pkgconfig \
        python3-dev \
        zeromq-dev \
        musl-dev \
        mercurial \
        libtool \
    && echo pip upgrade && pip3 install --upgrade pip \
    && echo python3.6 && cp /usr/bin/python3.6 /usr/bin/python \
    ## install Go
    && echo update chacher && apk --update-cache --allow-untrusted \
        --repository http://dl-4.alpinelinux.org/alpine/edge/community \
        --arch=x86_64 add \
        go \
    ## jupyter notebook
    && pip3 install  --ignore-installed pyzmq \
    && echo lns && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && echo pip3 install jupyter && pip3 install jupyter notebook

RUN apk --no-cache add \
    autoconf \
    automake \
    make


ENV GOPATH /go
    ## install gophernotes
RUN export GOPATH=/go \
    && go get -insecure github.com/pebbe/zmq4 \
    && go get -u github.com/gopherdata/gophernotes \
    && go install github.com/gopherdata/gophernotes \
    && cp /go/bin/gophernotes /usr/local/bin/ \
    && mkdir -p ~/.local/share/jupyter/kernels/gophernotes \
    && cp -r /go/src/github.com/gopherdata/gophernotes/kernel/* ~/.local/share/jupyter/kernels/gophernotes

    ## get the relevant Go packages
RUN  set -x \
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
    ## clean
    && find /usr/lib/python3.6 -name __pycache__ | xargs rm -r \
    && rm -rf \
        /root/.[acpw]* \
        ipaexg00301* \
    && rm -rf /var/cache/apk/*

EXPOSE 8888
CMD [ "jupyter", "notebook", "--no-browser", "--allow-root", "--debug", "--ip=0.0.0.0" ]
