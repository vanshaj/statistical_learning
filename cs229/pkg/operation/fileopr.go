package operation

import (
	"encoding/csv"
	"gonum.org/v1/gonum/mat"
	"log"
	"os"
	"strconv"
)

func ReadFile(filename string) (*mat.Dense, *mat.Dense) {
	xCols := 5
	log.Printf("Reading file %s\n", filename)
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	csvReader := csv.NewReader(file)
	record, err := csvReader.ReadAll()
	xFeatures := make([]float64, 0, len(record)*xCols) // 3 columns in the dataset
	yFeatures := make([]float64, 0, len(record)*1)     // 1 column in the dataset
	numFeatures := xCols
	for i := range len(record) {
		for j := range numFeatures { // as there are 3 columns
			if i == 0 {
				continue
			}
			val, err := strconv.ParseFloat(record[i][j], 64)
			if err != nil {
				//log.Fatalf("error parsing %s\n", err)
			}
			// At last column, we need to add 1.0 for the bias term and use the actual value as the label in matrix y
			if j == numFeatures-1 {
				xFeatures = append(xFeatures, 1.0)
				yFeatures = append(yFeatures, float64(val))
			} else {
				xFeatures = append(xFeatures, float64(val))
			}
		}
	}
	log.Printf("Converting file %s to matrix\n", filename)
	log.Printf("Size of array is %d, and dimension of matrix is %d*%d\n", len(xFeatures), len(yFeatures), xCols)
	x := mat.NewDense(len(yFeatures), xCols, xFeatures)
	y := mat.NewDense(len(yFeatures), 1, yFeatures)
	return x, y
}
