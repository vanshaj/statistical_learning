package app

import (
	"math"

	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/db"
	"github.com/vanshaj/statistical_learning/linear/internal"
)

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
