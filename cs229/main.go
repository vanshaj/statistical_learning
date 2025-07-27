package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/vanshaj/cs229/ps1"
)

func main() {
	fmt.Println("Regression YaYYYYY")
	filename := "./data/ps1/ds1_train.csv"
	absFilePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, filename)
	}
	l := ps1.LogisticRegression{}
	l.Fit(false, absFilePath)
}
