package notebookUtils

import (
	"os"
	"encoding/csv"
)

func LoadCsvDataset(path string) ([][]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return [][]string{}, err
	}
	csvFile := csv.NewReader(file)
	records, err := csvFile.ReadAll()
	if err != nil {
		return [][]string{}, err
	}
	file.Close()
	return records, nil
}

func TransposeStringMatrix(matrix [][]string) [][]string {
	r := make([][]string, len(matrix[0])) //39
	for index := range r {
		r[index] = make([]string, len(matrix))
	}
	for fv, f := range matrix {
		for cv, c := range f {
			r[cv][fv] = c
		}
	}
	return r
}
