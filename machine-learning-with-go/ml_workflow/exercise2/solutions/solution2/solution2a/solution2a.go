// All material is licensed under the Apache License Version 2.0, January 2004
// http://www.apache.org/licenses/LICENSE-2.0

// go build
// ./solution1

// Sample program to pre-process data for quality control of a regression model.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
)

// PredictionData includes the data necessary to make
// a prediction and encodes the output prediction.
type PredictionData struct {
	Prediction      float64          `json:"predicted_diabetes_progression,omitempty"`
	IndependentVars []IndependentVar `json:"independent_variables"`
	DependentVar    float64          `json:"dependent_variable"`
}

// IndependentVar include information about and a
// value for an independent variable.
type IndependentVar struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
}

func main() {

	// Declare the input and output directory/file flags.
	inFilePtr := flag.String("inFile", "", "The file containing the qc data.")
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

	// Loop of records in the CSV, pre-processing and saving them.
	var header []string
	for idx, record := range trainingData {

		// Collect the column names from the header.
		if idx == 0 {
			header = record
			continue
		}

		// Create a PredictionData value.
		var predictionData PredictionData

		// Fill the values in the predictionData value.
		var independentVars []IndependentVar
		for varID, value := range record {

			// Fill the observed value.
			if varID == 10 {

				// Parse the observed value.
				observation, err := strconv.ParseFloat(value, 64)
				if err != nil {
					log.Fatal(err)
				}

				// Add the observed value to predictionData.
				predictionData.DependentVar = observation
				continue
			}

			// Parse the feature value.
			feature, err := strconv.ParseFloat(value, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add the independent variable.
			independentVar := IndependentVar{
				Name:  header[varID],
				Value: feature,
			}

			independentVars = append(independentVars, independentVar)
		}

		predictionData.IndependentVars = independentVars

		// Save the pre-processed record.
		outputData, err := json.MarshalIndent(predictionData, "", "  ")
		if err != nil {
			log.Fatal(err)
		}

		// Save the marshalled output to a file.
		outputFile := strconv.Itoa(idx) + ".json"
		if err := ioutil.WriteFile(filepath.Join(*outDirPtr, outputFile), outputData, 0644); err != nil {
			log.Fatal(err)
		}
	}
}
