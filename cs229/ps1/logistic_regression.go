package ps1

import (
	"log"

	//"log"
	"math"

	"gonum.org/v1/gonum/mat"
)

type LogisticRegression struct {
	theta *mat.Dense
}

func (l *LogisticRegression) hypothesis(theta, x *mat.Dense) *mat.Dense {
	//log.Println("Calculating Hypothesis for LogisticRegression...")
	numExamples, _ := x.Dims()
	//log.Printf("Number of examples in the data set are %d, and number of features are %d\n", numExamples, numFeatures)
	ePow := mat.NewDense(numExamples, 1, nil)
	ePow.Mul(x, theta)
	y := mat.NewDense(numExamples, 1, nil)
	for i := range numExamples {
		denominator := 1 + math.Exp(-ePow.At(i, 0))
		value := 1 / denominator
		y.Set(i, 0, value)
	}
	//log.Printf("Calculated the Hypothesis values and dimensions are %d*1\n", numExamples)
	return y
}

func (l *LogisticRegression) gradient(theta, x, y *mat.Dense) *mat.Dense {
	//log.Println("Calculating gradient for the LogisticRegression...")
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

func (l *LogisticRegression) hessian(theta, x *mat.Dense) *mat.Dense {
	numSamples, numFeatures := x.Dims()
	hypoth := l.hypothesis(theta, x)
	dMatrix := mat.NewDense(numSamples, numSamples, nil)
	for i := range numSamples {
		dMatrix.Set(i, i, hypoth.At(i, 0)*(1-hypoth.At(i, 0)))
	}
	transposeX := x.T()
	firstPart := mat.NewDense(numFeatures, numSamples, nil)
	firstPart.Mul(transposeX, dMatrix)
	secondPart := mat.NewDense(numFeatures, numFeatures, nil)
	secondPart.Mul(firstPart, x)
	for i := range numFeatures {
		for j := range numFeatures {
			secondPart.Set(i, j, (1.0/float64(numSamples))*secondPart.At(i, j))
		}
	}
	return secondPart
}

func (l *LogisticRegression) fitUsingNewton(x, y *mat.Dense) *mat.Dense {
	_, numFeatures := x.Dims()
	theta := mat.NewDense(numFeatures, 1, nil)
	for range 100 {
		gradi := l.gradient(theta, x, y)
		hess := l.hessian(theta, x)
		hessInv := mat.NewDense(numFeatures, numFeatures, nil)
		err := hessInv.Inverse(hess)
		if err != nil {
			panic(err)
		}
		gradiMul := mat.NewDense(numFeatures, 1, nil)
		gradiMul.Mul(hessInv, gradi)
		theta.Sub(theta, gradiMul)
	}
	return theta
}

func (l *LogisticRegression) fitGradient(x, y *mat.Dense) *mat.Dense {
	learningRate := 0.001
	_, numFeatures := x.Dims()
	theta := mat.NewDense(numFeatures, 1, nil)
	for range 1000000 {
		gradi := l.gradient(theta, x, y)
		gradiMul := mat.NewDense(numFeatures, 1, nil)
		for i := range numFeatures {
			value := learningRate * gradi.At(i, 0)
			gradiMul.Set(i, 0, value)
		}
		theta.Sub(theta, gradiMul)
	}
	return theta
}

func (l *LogisticRegression) Fit(isNewton bool, x, y *mat.Dense) {
	if isNewton {
		log.Println("Using newton method")
		l.theta = l.fitUsingNewton(x, y)
		log.Printf("Output vector = %v\n", mat.Formatted(l.theta, mat.Prefix("    "), mat.Squeeze()))
	} else {
		log.Println("Using gradient descent")
		l.theta = l.fitGradient(x, y)
		log.Printf("Output vector = %v\n", mat.Formatted(l.theta, mat.Prefix("    "), mat.Squeeze()))
	}
}

func (l *LogisticRegression) Predict(x *mat.Dense) {
	log.Println("Predicting using Logistic Regression...")
	hypothesis := l.hypothesis(l.theta, x)
	numExamples, _ := x.Dims()
	predictions := mat.NewDense(numExamples, 1, nil)
	for i := range numExamples {
		if hypothesis.At(i, 0) >= 0.5 {
			predictions.Set(i, 0, 1)
		} else {
			predictions.Set(i, 0, 0)
		}
	}
	log.Printf("Output vector = %v\n", mat.Formatted(predictions, mat.Prefix("    "), mat.Squeeze()))
}
