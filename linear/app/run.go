package app

import (
	"fmt"
	"math"

	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/db"
	"github.com/vanshaj/statistical_learning/linear/internal"
)

var (
	CREATE_BOSTON_TABLE_QUERY      string = fmt.Sprintf("CREATE TABLE IF NOT EXISTS BOSTON AS SELECT * FROM read_csv(%s)", "data/boston.csv")
	SELECT_LSTATE_AGE_INTERACTION  string = "SELECT MULTIPLY(LSTAT, AGE) FROM BOSTON"
	SELECT_LSTATE_POLY_INTERACTION string = "SELECT MULTIPLY(LSTAT, LSTAT) FROM BOSTON"
	SELECT_CRIM_QUERY              string = "SELECT CRIM FROM BOSTON"
	SELECT_ZN_QUERY                string = "SELECT ZN FROM BOSTON"
	SELECT_INDUS_QUERY             string = "SELECT INDUS FROM BOSTON"
	SELECT_CHAS_QUERY              string = "SELECT CHAS FROM BOSTON"
	SELECT_NOX_QUERY               string = "SELECT NOX FROM BOSTON"
	SELECT_RM_QUERY                string = "SELECT RM FROM BOSTON"
	SELECT_AGE_QUERY               string = "SELECT AGE FROM BOSTON"
	SELECT_DIS_QUERY               string = "SELECT DIS FROM BOSTON"
	SELECT_RAD_QUERY               string = "SELECT RAD FROM BOSTON"
	SELECT_TAX_QUERY               string = "SELECT TAX FROM BOSTON"
	SELECT_PTRATIO_QUERY           string = "SELECT PTRATIO FROM BOSTON"
	SELECT_LSTAT_QUERY             string = "SELECT LSTAT FROM BOSTON"
	SELECT_MEDV_QUERY              string = "SELECT MEDV FROM BOSTON"
)

func Run() error {
	if err := db.NewDuckDB(); err != nil {
		return err
	}
	defer db.CloseDB()
	// /*
	// 	_, err := calculateMultipleRegressionCoefficient([]string{SELECT_CRIM_QUERY, SELECT_ZN_QUERY,
	// 		SELECT_LSTAT_QUERY, SELECT_AGE_QUERY,
	// 		SELECT_INDUS_QUERY, SELECT_CHAS_QUERY,
	// 		SELECT_NOX_QUERY, SELECT_RM_QUERY,
	// 		SELECT_DIS_QUERY, SELECT_RAD_QUERY,
	// 		SELECT_TAX_QUERY, SELECT_PTRATIO_QUERY}, SELECT_MEDV_QUERY)
	// */
	_, err := calculateMultipleRegressionCoefficient([]string{SELECT_LSTAT_QUERY, SELECT_AGE_QUERY, SELECT_LSTATE_AGE_INTERACTION}, SELECT_MEDV_QUERY)
	// _, err = calculateMultipleRegressionCoefficient([]string{SELECT_LSTAT_QUERY, SELECT_AGE_QUERY, SELECT_LSTATE_POLY_INTERACTION}, SELECT_MEDV_QUERY)
	if err != nil {
		return err
	}
	return nil
}

func RunY() error {
	if err := db.NewDuckDB(); err != nil {
		return err
	}
	defer db.CloseDB()
	l, err := coefficientCalculation(SELECT_LSTAT_QUERY, SELECT_MEDV_QUERY)
	if err != nil {
		return err
	}
	newXs := []float64{5, 10, 15}
	newYs, _ := predict(newXs, l)
	for _, val := range newYs {
		log.Debug().Msgf("Values : %f\n", val)
	}
	rss, err := calculateRSS(SELECT_MEDV_QUERY, SELECT_LSTAT_QUERY, l)
	if err != nil {
		return err
	}
	rse := calculateRSE(rss, 506)
	stddev := math.Pow(rse, 2)
	stdBeta1, err := calculateBeta1StdError(stddev, SELECT_LSTAT_QUERY, l)
	if err != nil {
		return err
	}
	stdBeta0, err := calculateBeta0StdError(stddev, SELECT_LSTAT_QUERY, l)
	if err != nil {
		return err
	}
	ci_slope, _ := CI_Slope(l, stdBeta1)
	ci_intercept, _ := CI_Intercept(l, stdBeta0)
	fmt.Printf("CI slope is %f\n", ci_slope)
	fmt.Printf("CI intercept %f\n", ci_intercept)
	predictCI(newXs, ci_slope, ci_intercept)
	// for _, val := range ci_interval {
	// 	log.Error().Msgf("Confidence interval %f\n", val)
	// }
	return nil
}

func RunX() error {
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
	err = internal.PlotLR(SELECT_LSTAT_QUERY, SELECT_MEDV_QUERY, l)
	if err != nil {
		return err
	}
	return nil
}
