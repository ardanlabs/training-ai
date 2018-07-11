package notebookUtils

import (
	"os"
	"io/ioutil"
)

func GetGraph(graphName string) ([]byte, error) {
	infile, err := os.Open(graphName)
	bytes, err := ioutil.ReadAll(infile)
	infile.Close()
	return bytes, err
}
