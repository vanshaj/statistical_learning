package ps1

import (
	"log"
	"math"

	"gonum.org/v1/gonum/mat"
)

type PoissonRegression struct {
	theta *mat.Dense
}

func (p *PoissonRegression) hypothesis(theta, x *mat.Dense) *mat.Dense {
	numExamples, _ := x.Dims()
	ePow := mat.NewDense(numExamples, 1, nil)
	// m*n * n*1
	ePow.Mul(x, theta)
	y := mat.NewDense(numExamples, 1, nil)
	for i := range numExamples {
		value := math.Exp(ePow.At(i, 0))
		y.Set(i, 0, value)
	}
	return y
}

func (p *PoissonRegression) gradient(theta, x, y *mat.Dense) *mat.Dense {
	hypothesis := p.hypothesis(theta, x)
	numExamples, numFeatures := x.Dims()
	errorMLE := mat.NewDense(numExamples, 1, nil)
	errorMLE.Sub(y, hypothesis)
	trasposeX := x.T()
	gradientTerm := mat.NewDense(numFeatures, 1, nil)
	gradientTerm.Mul(trasposeX, errorMLE)
	finalGradeint := mat.NewDense(numFeatures, 1, nil)
	for i := range numFeatures {
		value := (-1.0 / float64(numExamples)) * gradientTerm.At(i, 0)
		finalGradeint.Set(i, 0, value)
	}
	return finalGradeint
}

func (p *PoissonRegression) fitGradient(x, y *mat.Dense) *mat.Dense {
	learningRate := 2e-7
	_, numFeatures := x.Dims()
	theta := mat.NewDense(numFeatures, 1, nil)
	for range 100000 {
		gradi := p.gradient(theta, x, y)
		gradiMul := mat.NewDense(numFeatures, 1, nil)
		for i := range numFeatures {
			value := learningRate * gradi.At(i, 0)
			gradiMul.Set(i, 0, value)
		}
		theta.Sub(theta, gradiMul)
	}
	return theta
}

func (p *PoissonRegression) Fit(x, y *mat.Dense) {
	log.Println("Using gradient descent")
	p.theta = p.fitGradient(x, y)
	log.Printf("Output vector = %v\n", mat.Formatted(p.theta, mat.Prefix("    "), mat.Squeeze()))
}

func (p *PoissonRegression) Predict(x *mat.Dense) {
	log.Println("Predicting using Poisson Regression...")
	predictions := p.hypothesis(p.theta, x)
	log.Printf("Output vector = %.6f\n", mat.Formatted(predictions, mat.Prefix("    "), mat.Squeeze()))
}
