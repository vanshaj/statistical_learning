package app

import (
	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/db"
	"github.com/vanshaj/statistical_learning/linear/internal"
)

/*
the first column will be initiliazed with 1
Then each column will be initiliazed with the column values of that variable
*/
func calculateMultipleRegressionCoefficient(x_query []string, y_query string) (*internal.MultipleCoeff, error) {
	yData, err := db.GetValues(y_query)
	if err != nil {
		return nil, err
	}
	// Number of rows will be the number of response values
	xData := make([][]float64, len(yData))
	for i, cur_query := range x_query {
		value, err := db.GetValues(cur_query)
		if err != nil {
			return nil, err
		}
		for j := 0; j < len(value); j++ {
			if i == 0 {
				// Number of cols for each row will be number of predictors + 1 as first column will be initiliazed with 1
				xData[j] = make([]float64, len(x_query)+1)
				xData[j][0] = 1
			}
			xData[j][i+1] = value[j]
		}
	}
	m := internal.NewMultipleCoeff()
	xSingData := make([]float64, len(yData)*(len(x_query)+1))
	var count int
	for i := 0; i < len(yData); i++ {
		for j := 0; j < len(x_query)+1; j++ {
			xSingData[count] = xData[i][j]
			count++
		}
	}
	m.CalculateCoefficient(xSingData, yData, len(yData), len(x_query)+1)
	for _, val := range m.Coefficients {
		log.Debug().Msgf("Coefficients predicted are %f\n", val)
	}
	return m, nil
}
