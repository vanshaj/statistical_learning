package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"strings"

	_ "github.com/marcboeker/go-duckdb"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

var (
	FILENAME             string = "College.csv"
	CREATE_COLLEGE_QUERY string = fmt.Sprintf("CREATE TABLE IF NOT EXISTS COLLEGE AS select * from read_csv(%s)", FILENAME)
	CREATE_AUTO_QUERY    string = fmt.Sprintf("CREATE TABLE IF NOT EXISTS AUTO AS select * from read_csv(%s)", FILENAME)
)

func main() {
	query := "select accept, enroll, top10perc from college where top10perc > 50"
	matrix := dbToMat(query)
	val := matrix.At(2, 2)
	fmt.Println(val)
	mean_std()
	matrix2 := dbToMat("select horsepower, mpg from auto")
	plotXY(matrix2, "horsepower", "mpg")

	// plot TSS vs fitted values
	query = "select mpg - (select avg(mpg) from auto) as tss, mpg from auto"
	matrix_tss_y := dbToMat(query)
	plotXY(matrix_tss_y, "fitted_value", "tss")
	query = "select mpg - (select avg(mpg) from auto) as tss, mpg from auto"
}

func plotXY(mat *mat.Dense, xColName string, yColName string) {
	rows := mat.RawMatrix().Rows
	pts := make(plotter.XYs, rows)
	for j := 0; j < rows; j++ {
		pts[j].X = mat.At(j, 0)
	}
	for j := 0; j < rows; j++ {
		pts[j].Y = mat.At(j, 1)
	}
	// Create a new Plot
	p := plot.New()
	p.Title.Text = fmt.Sprintf("%s Vs %s", strings.ToUpper(yColName), strings.ToUpper(xColName))
	p.X.Label.Text = xColName
	p.Y.Label.Text = yColName
	// Add grid to the plot
	p.Add(plotter.NewGrid())
	// Create a new scatter with the plotter.Xys
	s, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err.Error())
	}
	// Add scatter plotter to the plot
	p.Add(s)
	// Save the plot
	err = p.Save(400, 200, fmt.Sprintf("%sVS%s.png", yColName, xColName))
	if err != nil {
		log.Fatal(err.Error())
	}
}

func createAuto() {
	db, err := sql.Open("duckdb", "test.db")
	if err != nil {
		log.Fatalf("Failed to open database due to error %s\n", err)
	}
	defer db.Close()
	_, err = db.Exec(CREATE_AUTO_QUERY)
	if err != nil {
		log.Fatalf("Failed to execute query due to error %s\n", err)
	}
}

func createCollege() {
	ctx := context.Background()
	db, err := sql.Open("duckdb", "test.db")
	if err != nil {
		log.Fatalf("Failed to open database due to error %s\n", err)
	}
	defer db.Close()
	conn, err := db.Conn(ctx)
	if err != nil {
		log.Fatalf("Failed to create connection due to error %s\n", err)
	}
	defer conn.Close()
	rows, err := conn.QueryContext(ctx, CREATE_COLLEGE_QUERY)
	if err != nil {
		log.Fatalf("Failed to execute query due to error %s\n", err)
	}
	rows.Close()
}

func mean_std() {
	cols := []string{
		"mpg",
		"cylinders",
		"displacement",
		"horsepower",
		"weight",
		"acceleration",
	}
	db, err := sql.Open("duckdb", "test.db")
	if err != nil {
		log.Fatalf("Failed to open database due to error %s\n", err)
	}
	defer db.Close()
	stddev_query := "select stddev(%s) from (select %s from auto offset 10 limit 75) as subset"
	mean_query := "select mean(%s) from (select %s from auto offset 10 limit 75) as subset"
	// Then iterate over the slice of columns
	for _, col := range cols {
		// Get the column series
		var stddev float64
		var mean float64
		row := db.QueryRow(fmt.Sprintf(stddev_query, col, col))
		err = row.Scan(&stddev)
		if err != nil {
			log.Fatalf("failed to exec query due to error %s", err)
		}
		row = db.QueryRow(fmt.Sprintf(mean_query, col, col))
		err = row.Scan(&mean)
		fmt.Printf("Std of %s is %f and mean is %f\n", col, stddev, mean)
	}

}

// where can be like top10perc > 50
func dbToMat(query string) *mat.Dense {
	db, err := sql.Open("duckdb", "test.db")
	if err != nil {
		log.Fatalf("Failed to open database due to error %s\n", err)
	}
	defer db.Close()
	items := make([]float64, 0)
	rowsData, err := db.QueryContext(context.Background(), query)
	if err != nil {
		log.Fatalf("Unable to get data %s\n", err)
	}
	defer rowsData.Close()
	columns, err := rowsData.Columns()
	if err != nil {
		log.Fatalf("Unable to get columns from query due to error %s\n", err)
	}
	c := len(columns)
	var r int
	for rowsData.Next() {
		item := make([]float64, c)
		itemPtr := make([]interface{}, c)
		for i := range item {
			itemPtr[i] = &item[i]
		}
		rowsData.Scan(itemPtr...)
		items = append(items, item...)
		r++
	}
	mat := mat.NewDense(r, c, items)
	return mat
}
