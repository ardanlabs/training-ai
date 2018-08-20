all: compile docker push clean

compile:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o infer

docker:
	docker build --force-rm=true -t gopherdata/gc2018:inference .

push:
	docker push gopherdata/gc2018:inference

clean:
	rm infer
