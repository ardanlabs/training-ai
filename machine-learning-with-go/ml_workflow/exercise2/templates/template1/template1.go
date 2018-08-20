// All material is licensed under the Apache License Version 2.0, January 2004
// http://www.apache.org/licenses/LICENSE-2.0

// go build
// ./solution1

// Sample program to train a regression model with multiple independent variables.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
)

// ModelInfo includes the information about the
// model that is output from the training.
type ModelInfo struct {
	Intercept    float64           `json:"intercept"`
	Coefficients []CoefficientInfo `json:"coefficients"`
}

// CoefficientInfo include information about a
// particular model coefficient.
type CoefficientInfo struct {
	Name        string  `json:"name"`
	Coefficient float64 `json:"coefficient"`
}

func main() {

	// Declare the input and output directory/file flags.
	inFilePtr := flag.String("inFile", "", "The file containing the training data.")
	outDirPtr := flag.String("outDir", "", "The output directory")

	// Parse the command line flags.
	flag.Parse()

	// Open the training dataset file.
	f, err := os.Open(*inFilePtr)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)

	// Read in all of the CSV records
	reader.FieldsPerRecord = 11
	trainingData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Create the value(s) needed to train a model using
	// github.com/sajari/regression or gonum.

	// Train/fit the regression model similar to how we did it
	// in our exploratory notebook.

	// Fill in the model information into a model info struct.
	modelInfo := ModelInfo{
		Intercept: r.Coeff(0),
		Coefficients: []CoefficientInfo{
			CoefficientInfo{
				Name: "bmi",
				// Coefficient: ?,
			},
			CoefficientInfo{
				Name: "ltg",
				// Coefficient: ?,
			},
		},
	}

	// Marshal the model information.
	outputData, err := json.MarshalIndent(modelInfo, "", "  ")
	if err != nil {
		log.Fatal(err)
	}

	// Save the marshalled output to a file.
	if err := ioutil.WriteFile(filepath.Join(*outDirPtr, "model.json"), outputData, 0644); err != nil {
		log.Fatal(err)
	}
}
