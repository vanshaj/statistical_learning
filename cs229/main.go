package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"github.com/vanshaj/cs229/ps1"
	"gonum.org/v1/gonum/mat"
)

func readFile(filename string) (*mat.Dense, *mat.Dense) {
	log.Printf("Reading file %s\n", filename)
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	csvReader := csv.NewReader(file)
	xFeatures := make([]float64, 0, 2400)
	yFeatures := make([]float64, 0, 800)
	record, err := csvReader.ReadAll()
	numFeatures := 3
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
	log.Printf("Size of array is %d, and dimension of matrix is %d*%d\n", len(xFeatures), len(yFeatures), 3)
	x := mat.NewDense(len(yFeatures), 3, xFeatures)
	y := mat.NewDense(len(yFeatures), 1, yFeatures)
	return x, y
}

func main() {
	fmt.Println("Regression YaYYYYY")
	filename := "./data/ps1/ds1_train.csv"
	absFilePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, filename)
	}
	l := ps1.LogisticRegression{}
	x, y := readFile(absFilePath)
	l.Fit(true, x, y)
	l.Fit(false, x, y)
}
