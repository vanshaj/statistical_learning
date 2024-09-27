package app

import (
	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/db"
	"github.com/vanshaj/statistical_learning/linear/internal"
)

func calculateMultipleRegressionCoefficient(x_query []string, y_query string) (*internal.MultipleCoeff, error) {
	xData := make([]float64, 0)
	for i, cur_query := range x_query {
		value, err := db.GetValues(cur_query)
		if err != nil {
			return nil, err
		}
		if i == 0 {
			one_slice := make([]float64, len(value))
			for j := 0; j < len(value); j++ {
				one_slice[j] = 1
			}
			xData = append(xData, one_slice...)
		}
		xData = append(xData, value...)
	}
	yData, err := db.GetValues(y_query)
	if err != nil {
		return nil, err
	}
	m := internal.NewMultipleCoeff()
	m.CalculateCoefficient(xData, yData, len(yData), len(x_query)+1)
	for _, val := range m.Coefficients {
		log.Debug().Msgf("Coefficients predicted are %f\n", val)
	}
	return m, nil
}
