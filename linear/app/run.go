package app

import (
	"fmt"
	"math"

	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/db"
)

var (
	CREATE_BOSTON_TABLE_QUERY string = fmt.Sprintf("CREATE TABLE IF NOT EXISTS BOSTON AS SELECT * FROM read_csv(%s)", "data/boston.csv")
	SELECT_LSTAT_QUERY        string = "SELECT LSTAT FROM BOSTON"
	SELECT_MEDV_QUERY         string = "SELECT MEDV FROM BOSTON"
)

func Run() error {
	if err := db.NewDuckDB(); err != nil {
		return err
	}
	defer db.CloseDB()
	l, err := coefficientCalculation(SELECT_LSTAT_QUERY, SELECT_MEDV_QUERY)
	if err != nil {
		return err
	}
	rss, err := calculateRSS(SELECT_MEDV_QUERY, SELECT_LSTAT_QUERY, l)
	if err != nil {
		return err
	}
	rse := calculateRSE(rss, 506)
	stddev := math.Pow(rse, 2)
	beta1StdError, err := calculateBeta1StdError(stddev, SELECT_LSTAT_QUERY, l)
	log.Debug().Msgf("Standard Error of Beta 1 is %f\n", beta1StdError)
	return nil
}
