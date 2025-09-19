package prediction

import (
	"fmt"
	"github.com/vanshaj/cs229/pkg/operation"
	"github.com/vanshaj/cs229/pkg/ps1"
	"log"
	"path/filepath"
)

func PredictPoisson() {
	fmt.Println("Regression YaYYYYY")
	filename := "./data/ps1/ds4_train.csv"
	absFilePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, filename)
	}
	p := ps1.PoissonRegression{}
	x, y := operation.ReadFile(absFilePath)
	p.Fit(x, y)

	predictFilename := "./data/ps1/ds4_valid_copy.csv"
	absFilePathPredict, err := filepath.Abs(predictFilename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, predictFilename)
	}
	x_predict, _ := operation.ReadFile(absFilePathPredict)
	p.Predict(x_predict)
}
