package db

import (
	"context"
	"database/sql"

	_ "github.com/marcboeker/go-duckdb"
	"github.com/rs/zerolog/log"
)

type dB struct {
	*sql.DB
}

var DB *dB

func NewDuckDB() error {
	db, err := sql.Open("duckdb", "test.duckdb")
	if err != nil {
		log.Error().Msgf("Unable to open db connection to test.duckdb due to %s\n", err.Error())
		return err
	}
	DB = &dB{db}
	return nil
}

func CloseDB() error {
	if err := DB.Close(); err != nil {
		log.Error().Msgf("Unable to open db connection to test.duckdb due to %s\n", err.Error())
		return err
	}
	return nil
}

func GetValues(query string) ([]float64, error) {
	rowsData, err := DB.QueryContext(context.Background(), query)
	if err != nil {
		log.Error().Msgf("Unable to get lstatData due to %s\n", err.Error())
		return nil, err
	}
	items := make([]float64, 0)
	for rowsData.Next() {
		var item float64
		rowsData.Scan(&item)
		items = append(items, item)
	}
	return items, nil
}
