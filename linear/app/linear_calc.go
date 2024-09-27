package app

import (
	"fmt"
	"math"

	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/db"
	"github.com/vanshaj/statistical_learning/linear/internal"
)

func predictCI(Xs []float64, ci_slope []float64, ci_intercept []float64) {
	rows := len(Xs)
	columns := 2
	Ys := make([][]float64, rows)
	for i := range Ys {
		Ys[i] = make([]float64, columns)
	}
	for i, val := range Xs {
		Ys[i][0] = ci_intercept[0] + ci_slope[0]*val
	}
	for i, val := range Xs {
		Ys[i][1] = ci_intercept[1] + ci_slope[1]*val
	}
	for _, row := range Ys {
		fmt.Println(row)
	}
}

func predict(Xs []float64, l *internal.LinearCoeff) ([]float64, error) {
	Ys := make([]float64, len(Xs))
	for i, val := range Xs {
		Ys[i] = l.Intercept + l.Slope*val
	}
	return Ys, nil
}

func CI_Slope(l *internal.LinearCoeff, seBeta1 float64) ([]float64, error) {
	tvalue := 2.011
	errVal := float64(tvalue) * seBeta1
	ci := make([]float64, 2)
	ci[0] = l.Slope - errVal
	ci[1] = l.Slope + errVal
	return ci, nil
}

func CI_Intercept(l *internal.LinearCoeff, seBeta0 float64) ([]float64, error) {
	tvalue := 2.011
	errVal := float64(tvalue) * seBeta0
	ci := make([]float64, 2)
	ci[0] = l.Intercept - errVal
	ci[1] = l.Intercept + errVal
	return ci, nil
}

func coefficientCalculation(x_query, y_query string) (*internal.LinearCoeff, error) {
	/*
		_, err := db.DB.Exec(CREATE_BOSTON_TABLE_QUERY)
		if err != nil {
			log.Error().Msgf("Unable to create boston table due to %s\n", err.Error())
			return err
		}
	*/
	lstatData, err := db.GetValues(x_query)
	if err != nil {
		return nil, err
	}
	medvData, err := db.GetValues(y_query)
	if err != nil {
		return nil, err
	}

	l := internal.NewLinearCoeff()
	l.CalculateSlope(lstatData, medvData)
	l.CalculateIntercept(lstatData, medvData)

	log.Debug().Msgf("Intercept predicted is %f\n", l.Intercept)
	log.Debug().Msgf("Slope predicted is %f\n", l.Slope)
	return l, nil
}

func calculateRSS(y_query, x_query string, l *internal.LinearCoeff) (float64, error) {
	medvData, err := db.GetValues(y_query)
	if err != nil {
		return 0, err
	}
	lstatData, err := db.GetValues(x_query)
	if err != nil {
		return 0, err
	}

	rss := l.CalculateRSS(lstatData, medvData)
	return rss, nil
}

func calculateRSE(rss float64, n float64) float64 {
	return math.Sqrt((rss / (n - 2)))
}

func calculateBeta1StdError(stddev float64, x_query string, l *internal.LinearCoeff) (float64, error) {
	lstatData, err := db.GetValues(x_query)
	if err != nil {
		return 0, err
	}
	beta1StdError := l.CalculateStdErrorSlope(lstatData, stddev)
	return beta1StdError, nil
}

func calculateBeta0StdError(stddev float64, x_query string, l *internal.LinearCoeff) (float64, error) {
	lstatData, err := db.GetValues(x_query)
	if err != nil {
		return 0, err
	}
	beta0StdError := l.CalculateStdErrorIntercept(lstatData, stddev)
	return beta0StdError, nil
}
