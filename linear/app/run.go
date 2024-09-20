package app

import "github.com/vanshaj/statistical_learning/linear/db"

func Run() error {
	if err := db.NewDuckDB(); err != nil {
		return err
	}
	defer db.CloseDB()
	return coefficientCalculation()
}
