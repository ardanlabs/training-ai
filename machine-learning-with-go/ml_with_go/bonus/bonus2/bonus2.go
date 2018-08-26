package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"strconv"

	"gocv.io/x/gocv"
)

// readDescriptions reads the descriptions from a file
// and returns a slice of its lines.
func readDescriptions(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}

func main() {
	if len(os.Args) < 4 {
		fmt.Println("How to run:\ntf-classifier [camera ID] [modelfile] [descriptionsfile]")
		return
	}

	// Parse the command line arguments.
	deviceID, err := strconv.Atoi(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	model := os.Args[2]

	descriptions, err := readDescriptions(os.Args[3])
	if err != nil {
		log.Fatal(err)
	}

	// Open the default capture device, 0.
	webcam, err := gocv.VideoCaptureDevice(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	window := gocv.NewWindow("Tensorflow Classifier")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	// Read in the TensorFlow model.
	net := gocv.ReadNetFromTensorflow(model)
	defer net.Close()

	status := "Ready"
	statusColor := color.RGBA{0, 255, 0, 0}
	fmt.Printf("Start reading camera device: %v\n", deviceID)

	// Begin reading in frames from the camera.
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Error cannot read device %d\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// Convert the image Mat to 224x244 blob that the classifier can analyze.
		blob := gocv.BlobFromImage(img, 1.0, image.Pt(224, 244), gocv.NewScalar(0, 0, 0, 0), true, false)
		defer blob.Close()

		// Feed the blob into the classifier.
		net.SetInput(blob, "input")

		// Run a forward pass thru the network.
		prob := net.Forward("softmax2")
		defer prob.Close()

		// Reshape the results into a 1x1000 matrix.
		probMat := prob.Reshape(1, 1)
		defer probMat.Close()

		// Determine the most probable classification.
		_, maxVal, _, maxLoc := gocv.MinMaxLoc(probMat)

		// Display the classification.
		desc := "Unknown"
		if maxLoc.X < 1000 {
			desc = descriptions[maxLoc.X]
		}
		status = fmt.Sprintf("description: %v, maxVal: %v\n", desc, maxVal)
		gocv.PutText(&img, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
