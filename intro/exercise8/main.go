package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"

	_ "github.com/marcboeker/go-duckdb"
)

var (
	FILENAME     string = "College.csv"
	CREATE_QUERY string = fmt.Sprintf("CREATE TABLE COLLEGE AS select * from read_from_csv(%s)", FILENAME)
)

func createTable() {
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
	rows, err := conn.QueryContext(ctx, CREATE_QUERY)
	if err != nil {
		log.Fatalf("Failed to execute query due to error %s\n", err)
	}
	rows.Close()
}

func getCountTopCollege() {
	ctx := context.Background()
	select_query := "select count(*) from college where Top10perc > 50"
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
	rows, err := conn.QueryContext(ctx, select_query)
	if err != nil {
		log.Fatalf("Failed to execute query due to error %s\n", err)
	}
	defer rows.Close()
	var count int
	for rows.Next() {
		if err := rows.Scan(&count); err != nil {
			log.Fatalf("Unable to get the count of the value due to error %s\n", err)
		}
	}
	log.Printf("Number of rows are %d\n", count)

}

func main() {
	createTable()
	getCountTopCollege()
}
