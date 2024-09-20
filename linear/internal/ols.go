package internal

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/stat"
)

type LinearCoeff struct {
	Intercept float64
	Slope     float64
}

func NewLinearCoeff() *LinearCoeff {
	return &LinearCoeff{}
}

func (l *LinearCoeff) CalculateSlope(x, y []float64) error {
	if len(x) != len(y) {
		return errors.New("len of x and y doesnot match")
	}
	y_mean := stat.Mean(y, nil)
	x_mean := stat.Mean(x, nil)
	length := len(x)

	var numerator float64
	var denominator float64

	for i := 0; i < length; i++ {
		value := (x[i] - x_mean) * (y[i] - y_mean)
		numerator = numerator + value
	}

	for i := 0; i < length; i++ {
		value := math.Pow((x[i] - x_mean), 2)
		denominator = denominator + value
	}
	l.Slope = numerator / denominator
	return nil
}

func (l *LinearCoeff) CalculateIntercept(x, y []float64) error {
	if l.Slope == 0 {
		return errors.New("slope not identified")
	}
	y_mean := stat.Mean(y, nil)
	x_mean := stat.Mean(x, nil)

	l.Intercept = y_mean - l.Slope*x_mean

	return nil
}
