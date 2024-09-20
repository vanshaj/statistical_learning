package app

import (
	"fmt"

	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/db"
	"github.com/vanshaj/statistical_learning/linear/internal"
)

var (
	CREATE_BOSTON_TABLE_QUERY string = fmt.Sprintf("CREATE TABLE IF NOT EXISTS BOSTON AS SELECT * FROM read_csv(%s)", "data/boston.csv")
	SELECT_LSTAT_QUERY        string = "SELECT LSTAT FROM BOSTON"
	SELECT_MEDV_QUERY         string = "SELECT MEDV FROM BOSTON"
)

func coefficientCalculation() error {
	/*
		_, err := db.DB.Exec(CREATE_BOSTON_TABLE_QUERY)
		if err != nil {
			log.Error().Msgf("Unable to create boston table due to %s\n", err.Error())
			return err
		}
	*/
	lstatData, err := db.GetValues(SELECT_LSTAT_QUERY)
	if err != nil {
		return err
	}
	medvData, err := db.GetValues(SELECT_MEDV_QUERY)
	if err != nil {
		return err
	}

	l := internal.NewLinearCoeff()
	l.CalculateSlope(lstatData, medvData)
	l.CalculateIntercept(lstatData, medvData)

	log.Debug().Msgf("Intercept predicted is %f\n", l.Intercept)
	log.Debug().Msgf("Slope predicted is %f\n", l.Slope)
	return nil
}
