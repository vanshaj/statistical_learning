package main

import (
	"fmt"
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

func countEliteCollege() {
	filename := "College.csv"
	r, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Unable to open file '%s' due to error '%s'\n", filename, err.Error())
	}
	headerOption := dataframe.HasHeader(true)
	df := dataframe.ReadCSV(r, headerOption)
	spData := df.Filter(
		dataframe.F{
			Colname:    "Top10perc",
			Comparator: series.Greater,
			Comparando: 50,
		},
	)
	fmt.Println(spData.Nrow())
}
