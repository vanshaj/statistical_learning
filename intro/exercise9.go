package main

import (
	"fmt"
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)

func main() {
	dfa := ReadCsv("Auto.csv")
	cols := []string{
		"mpg",
		"cylinders",
		"displacement",
		"horsepower",
		"weight",
		"acceleration",
	}
	rows := dfa.Nrow()
	// Make a slice of the rows you want to select
	row_select := make([]int, 0, rows)
	for i := 0; i < rows; i++ {
		if i < 10 || i > 85 {
			row_select = append(row_select, i)
		}
	}
	// Pass the slice to Subset function to get the data frame
	df := dfa.Subset(row_select)

	// Then iterate over the slice of columns
	for _, col := range cols {
		// Get the column series
		colSeries := df.Col(col)
		// Now get the stats on the column series of a df
		stdCol := colSeries.StdDev()
		meanCol := colSeries.Mean()
		fmt.Printf("Std of %s is %f and mean is %f\n", col, stdCol, meanCol)
	}

}

func ReadCsv(filename string) dataframe.DataFrame {
	fr, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Unable to open file '%s' due to error '%s'\n", filename, err.Error())
	}
	headerOption := dataframe.HasHeader(true)
	df := dataframe.ReadCSV(fr, headerOption)
	return df
}

func dtM(df dataframe.DataFrame) *mat.Dense {
	r := df.Nrow()
	c := df.Ncol()
	itemCount := r * c
	items := make([]float64, itemCount)
	count := 0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			items[count] = df.Elem(i, j).Float()
			count++
		}
	}
	mat := mat.NewDense(r, c, items)
	return mat
}
