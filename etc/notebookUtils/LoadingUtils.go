package main

import (
	"os"
	"fmt"
	"image"
	"image/png"
	"image/color"
	"github.com/kniren/gota/dataframe"
	"strconv"
)

func main() {
	f, err := os.Open("/Users/ortegadiana/programacion/gophercon2018/confaidata/training-ai/machine-learning-with-go/ml_with_go/data/optdigits.tra")
	if err != nil {
		fmt.Println(err)
	}

	// Create a dataframe from the CSV file.
	// The types of the columns will be inferred.
	dataset := dataframe.ReadCSV(f)
	f.Close()
	fmt.Println(dataset.Dims())

	records := dataset.Records()

	for i, record := range records {
		if i > 0 {
			img := image.NewGray(image.Rect(0, 0, 8, 8))
			for j := 0; j < 64; j++ {
				va, _ := strconv.Atoi(record[j])
				img.Set(j%8, j/8, color.Gray{uint8(va * 16)})
			}
			f, _ := os.OpenFile("optImg/optdigit_"+strconv.Itoa(i)+"-"+ record[64]+".png", os.O_WRONLY|os.O_CREATE, 0600)
			png.Encode(f, img)
			f.Close()
		}
	}
}

	// Create an image
	/*img := image.NewGray16(image.Rect(0, 0, 8, 8))

	// Draw a red dot at (2, 3)
	img.Set(2, 3, color.RGBA{255, 0, 0, 255})

	// Save to out.png
	f, _ := os.OpenFile("out.png", os.O_WRONLY|os.O_CREATE, 0600)
	defer f.Close()
	png.Encode(f, img)*/
	// }
	// GetGraph returns the bytes corresponding to a
	// saved plot.
	/*func GetGraph(graphName string) ([]byte, error) {

		// Open the file.
		infile, err := os.Open(graphName)
		if err != nil {
			return nil, err
		}

		// Read in the contents of the file.
		bytes, err := ioutil.ReadAll(infile)
		if err != nil {
			return nil, err
		}

		// Close the file.
		infile.Close()

		return bytes, err
	}

	func main() {
		imgfile, err := os.Open("./img.jpg")

		if err != nil {
			fmt.Println("img.jpg file not found!")
			os.Exit(1)
		}

		defer imgfile.Close()

		img, err := png.Decode(imgfile)

		var data []gokmeans.Node

		var point []float64

		data = append(data, point)



		bounds := img.Bounds()

		fmt.Println(bounds)

		canvas := image.NewAlpha(bounds)

		// is this image opaque
		op := canvas.Opaque()

		fmt.Println(op)
	}*/
