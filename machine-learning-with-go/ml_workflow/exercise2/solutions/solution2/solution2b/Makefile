all: compile docker push clean

compile:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o qcontrol

docker:
	docker build --force-rm=true -t gopherdata/gc2018:qc .

push:
	docker push gopherdata/gc2018:qc

clean:
	rm qcontrol
