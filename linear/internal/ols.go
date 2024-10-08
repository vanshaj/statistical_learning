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

func (l *LinearCoeff) CalculateRSS(x, y []float64) float64 {
	var rss float64
	for i := 0; i < len(y); i++ {
		predicted_y := l.Intercept + l.Slope*x[i]
		rss = rss + math.Pow((y[i]-predicted_y), 2)
	}
	return rss
}

func (l *LinearCoeff) CalculateStdErrorSlope(x []float64, rseSquare float64) float64 {
	var denom float64
	mean_x := stat.Mean(x, nil)
	for i := 0; i < len(x); i++ {
		denom = denom + math.Pow((x[i]-mean_x), 2)
	}
	stderr := math.Sqrt((rseSquare / denom))
	return stderr
}

func (l *LinearCoeff) CalculateStdErrorIntercept(x []float64, rseSquare float64) float64 {
	part1 := 1 / len(x)
	var part2 float64
	part2_num := math.Pow(stat.Mean(x, nil), 2)
	var part2_den float64
	for _, val := range x {
		part2_den = part2_den + math.Pow((val-stat.Mean(x, nil)), 2)
	}
	part2 = part2_num / part2_den
	stderr := rseSquare * (float64(part1) + part2)
	return stderr
}
