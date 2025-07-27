package ps1

import (
	"encoding/csv"
	"log"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type LogisticRegression struct{}

func (l *LogisticRegression) hypothesis(theta, x *mat.Dense) *mat.Dense {
	log.Println("Calculating Hypothesis for LogisticRegression...")
	numExamples, numFeatures := x.Dims()
	log.Printf("Number of examples in the data set are %d, and number of features are %d\n", numExamples, numFeatures)
	ePow := mat.NewDense(numExamples, 1, nil)
	ePow.Mul(x, theta)
	y := mat.NewDense(numExamples, 1, nil)
	for i := range numExamples {
		denominator := 1 + math.Exp(-ePow.At(i, 0))
		value := 1 / denominator
		y.Set(i, 0, value)
	}
	log.Printf("Calculated the Hypothesis values and dimensions are %d*1\n", numExamples)
	return y
}

func (l *LogisticRegression) gradient(theta, x, y *mat.Dense) *mat.Dense {
	log.Println("Calculating gradient for the LogisticRegression...")
	hypothesis := l.hypothesis(theta, x)
	numExamples, numFeatures := x.Dims()
	errorMLE := mat.NewDense(numExamples, 1, nil)
	errorMLE.Sub(y, hypothesis)
	transposeX := x.T()
	gradientTerm := mat.NewDense(numFeatures, 1, nil)
	gradientTerm.Mul(transposeX, errorMLE)
	finalGradeint := mat.NewDense(numFeatures, 1, nil)
	for i := range numFeatures {
		value := (-1.0 / float64(numExamples)) * gradientTerm.At(i, 0)
		finalGradeint.Set(i, 0, value)
	}
	return finalGradeint
}

func (l *LogisticRegression) Hessian() {}

func (l *LogisticRegression) fitUsingNewton() {}

func (l *LogisticRegression) fitGradient(x, y *mat.Dense) *mat.Dense {
	learningRate := 0.0001
	_, numFeatures := x.Dims()
	theta := mat.NewDense(numFeatures, 1, nil)
	gradi := l.gradient(theta, x, y)
	for _ = range 8000 {
		gradiMul := mat.NewDense(numFeatures, 1, nil)
		for i := range numFeatures {
			value := learningRate * gradi.At(i, 0)
			gradiMul.Set(i, 0, value)
		}
		theta.Sub(theta, gradiMul)
	}
	return theta
}

func (l *LogisticRegression) Fit(isNewton bool, filename string) {
	log.Printf("Reading file %s\n", filename)
	file, err := os.Open(filename)
	if err != nil {
		log.Printf("Unable to open file %s\n", filename)
		return
	}
	csvReader := csv.NewReader(file)
	xFeatures := make([]float64, 0, 100000)
	yFeatures := make([]float64, 0, 1000)
	record, err := csvReader.ReadAll()
	log.Printf("Total number of examples are %d\n", len(record))
	for j := range 3 { // as there are 3 columns
		for i := range len(record) {
			if i == 0 {
				continue
			}
			val, err := strconv.ParseFloat(record[i][j], 64)
			if err != nil {
				log.Fatalf("error parsing %s\n", err)
			}
			if j == 3-1 {
				xFeatures = append(xFeatures, 0.0)
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
	/*
		log.Printf("Feature vector = %v\n", mat.Formatted(x, mat.Prefix("    "), mat.Squeeze()))
		log.Printf("Output vector = %v\n", mat.Formatted(y, mat.Prefix("    "), mat.Squeeze()))
	*/
	if isNewton {
		log.Println("Using newton method")
		// l.fitUsingNewton()
	} else {
		log.Println("Using gradient descent")
		theta := l.fitGradient(x, y)
		log.Printf("Output vector = %v\n", mat.Formatted(theta, mat.Prefix("    "), mat.Squeeze()))
	}
}
