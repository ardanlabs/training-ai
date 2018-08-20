all: compile docker push clean

compile:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o train

docker:
	docker build --force-rm=true -t gopherdata/gc2018:training .

push:
	docker push gopherdata/gc2018:training

clean:
	rm train
