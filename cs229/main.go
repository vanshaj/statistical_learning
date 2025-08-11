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

func predict_poissongr() {
	fmt.Println("Regression YaYYYYY")
	filename := "./data/ps1/ds4_train.csv"
	absFilePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, filename)
	}
	p := ps1.PoissonRegression{}
	x, y := readFile(absFilePath)
	p.Fit(x, y)

	predictFilename := "./data/ps1/ds4_valid_copy.csv"
	absFilePathPredict, err := filepath.Abs(predictFilename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, predictFilename)
	}
	x_predict, _ := readFile(absFilePathPredict)
	p.Predict(x_predict)

}

func predict_lgr() {
	fmt.Println("Regression YaYYYYY")
	filename := "./data/ps1/ds2_train.csv"
	absFilePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, filename)
	}
	l := ps1.LogisticRegression{}
	x, y := readFile(absFilePath)
	l.Fit(true, x, y)
	//l.Fit(false, x, y)
	predictFilename := "./data/ps1/ds2_valid_copy.csv"
	absFilePathPredict, err := filepath.Abs(predictFilename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, predictFilename)
	}
	x_predict, _ := readFile(absFilePathPredict)
	l.Predict(x_predict)
}

func predict_gda() {
	fmt.Println("Regression YaYYYYY")
	filename := "./data/ps1/ds2_train.csv"
	absFilePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, filename)
	}
	l := ps1.BinaryGda{}
	x, y := readFile(absFilePath)
	l.Fit(x, y)
	predictFilename := "./data/ps1/ds2_valid_copy.csv"
	absFilePathPredict, err := filepath.Abs(predictFilename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, predictFilename)
	}
	x_predict, _ := readFile(absFilePathPredict)
	l.Predict(x_predict)
}

func main() {
	predict_poissongr()
}
