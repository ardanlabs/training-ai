package main

import (
	"fmt"
	"log"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"os"
	"image"
	"io"
	"image/png"
	"encoding/json"
	"io/ioutil"
	"reflect"
)

const (
	batchSize = 1
	channels  = 3
)

type labelItem struct {
	Name        string `json:"name"`
	Id          int    `json:"id"`
	DisplayName string `json:"display_name"`
}

// img.At(x, y).RGBA() returns four uint32 values; we want a Pixel in uint8
func rgbaToPixel(r uint32, g uint32, b uint32, a uint32) []uint8 {
	return []uint8{uint8(r / 256), uint8(g / 256), uint8(b / 256)}
}

func loadImageAsTensor(filePath string) (*tf.Tensor, error) {
	// open path for image
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatal("an error opening file: ", err.Error())
	}

	// it is a png image, we decode it to read its information
	img, err := png.Decode(file)
	if err != nil {
		log.Fatal("error decoding png image", err)
	}

	// read bounds of image (we'll need that later)
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	file.Close()
	file, _ = os.Open(filePath)

	// define the variable that will hold a batch of images to transform in tensor. It has 4 dimensions: N R G B
	// where N is the quantity of images in the batch (1) and R G B is the color intensity of a pixel in the corresponding channel
	var imageData [][][][]uint8
	imageData = append(imageData, loadOneImageFromFile(filePath))

	lshape := []int64{int64(batchSize), int64(width), int64(height), int64(channels)}
	tf.ReadTensor(tf.Uint8, lshape, file)

	// return tensor with image data
	return tf.NewTensor(imageData)
}

//load an image from a file and transform it to [][][] uint8. each dimension represents a channel (RGB)
func loadOneImageFromFile(imageName string) [][][]uint8 {
	fmt.Println("opening", imageName)
	existingImageFile, err := os.Open(imageName)
	if err != nil {
		// Handle error
		log.Fatal("error loading image", err)
	}
	defer existingImageFile.Close()
	return getPixels(existingImageFile)
}

func getPixels(file io.Reader) [][][]uint8 {
	img, _, err := image.Decode(file)

	if err != nil {
		return nil
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var pixels [][][]uint8
	for y := 0; y < height; y++ {
		var row [][]uint8
		for x := 0; x < width; x++ {
			row = append(row, rgbaToPixel(img.At(x, y).RGBA()))
		}
		pixels = append(pixels, row)
	}

	return pixels
}

// Load saved model from folder
func loadModel(modeldir *string) (*tf.SavedModel, error) {
	if modeldir == nil {
		log.Println("error loading. Model must exist")
		return nil, nil
	}
	return tf.LoadSavedModel(*modeldir, []string{"serve"}, nil)
}

// function for predicting the objects found in an image .png
func Prediction(modelPath, imagePath, labelsPath string) {
	//load labels, we will use them later
	labels := readLabels(labelsPath)

	// load saved model
	model, err := loadModel(&modelPath)
	if err != nil {
		log.Println(err)
	}
	defer model.Session.Close()

	// create input tensor
	tensor, err := loadImageAsTensor(imagePath)
	if err != nil {
		log.Println(err)
	}
	tensors := map[tf.Output]*tf.Tensor{
		model.Graph.Operation("image_tensor").Output(0): tensor,
	}

	//create ouput tensor: call the operations where we'll get the answer of the model
	// this model returns: 1. a score in % of the detected element. 2. A class (label). It is a number. see labels file for understanding better.
	// 3. the coordinates of the bounding box of the detected object
	outputs := []tf.Output{
		model.Graph.Operation("detection_scores").Output(0),
		model.Graph.Operation("detection_classes").Output(0),
		model.Graph.Operation("detection_boxes").Output(0),
	}

	// run the model in a session, and get the result
	result, runErr := model.Session.Run(
		tensors,
		outputs,
		nil,
	)
	if runErr != nil {
		log.Fatal("error running the session with input, err:", runErr.Error())
		return
	}
	// parse the result to be human readable
	parseResult(result, labels)
}

/**
* Parse the result. result is a vector where each position contains the output of tensors we requested.
* in this case we have 3 outputs:
* result[0]= detection_scores
* result[1]= detection_classes
* result[2]= detection_boxes
*/
func parseResult(result []*tf.Tensor, labels []labelItem) {
	//each position of the result vector is of type interface{}, we know it is an array.
	//we must read values using reflection
	scores := reflect.ValueOf(result[0].Value()).Index(0)
	fmt.Println("scores", scores) //scores

	clases := reflect.ValueOf(result[1].Value()).Index(0)
	fmt.Println("clases", clases) //clases

	boxes := reflect.ValueOf(result[2].Value()).Index(0)
	fmt.Println("boxes", boxes) //bounding boxes

	for i := 0; i < 10; i++ {
		value := scores.Index(i)
		if value.Float() > 0.5 { // we only show things where the model is "sure", it means score > 50%
			item := findLabel(labels, int(clases.Index(i).Float()))
			fmt.Println("Detected", item.DisplayName, "with probability", value.Float())
		}
	}
}

// utility to find a labelItem from an id
func findLabel(labels []labelItem, id int) labelItem {
	for _, v := range labels {
		if int(v.Id) == id {
			return v
		}
	}
	return labelItem{}
}

// load Labels from file
func readLabels(labelsPath string) []labelItem {
	var items []labelItem
	jsonFile, err := os.Open(labelsPath)
	if err != nil {
		log.Fatal("labels error ", err)
		return items
	}
	byteValue, _ := ioutil.ReadAll(jsonFile)
	err = json.Unmarshal(byteValue, &items)
	if err != nil {
		log.Fatal("unmarshalling error", err)
		return items
	}
	return items
}

func main() {
	// WARNING!!! change to the path of your model, labels and image
	modelPath := "/home/model/training-ai/machine-learning-with-go/ml_with_go/data/ssd_mobilenet_v1_coco_2018_01_28/saved_model"
	labelsPath := "/home/ml/training-ai/machine-learning-with-go/ml_with_go/data/ssd_mobilenet_v1_coco_2018_01_28/labels.json"
	imagePath := "/home/notebooks/training-ai/machine-learning-with-go/ml_with_go/data/office.png"
	Prediction(modelPath, imagePath, labelsPath)
}
