package prediction

import (
	"fmt"
	"github.com/vanshaj/cs229/pkg/operation"
	"github.com/vanshaj/cs229/pkg/ps1"
	"log"
	"path/filepath"
)

func PredictLogistic() {
	fmt.Println("Regression YaYYYYY")
	filename := "./data/ps1/ds2_train.csv"
	absFilePath, err := filepath.Abs(filename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, filename)
	}
	l := ps1.LogisticRegression{}
	x, y := operation.ReadFile(absFilePath)
	l.Fit(true, x, y)
	//l.Fit(false, x, y)
	predictFilename := "./data/ps1/ds2_valid_copy.csv"
	absFilePathPredict, err := filepath.Abs(predictFilename)
	if err != nil {
		log.Fatalf("Unable to generate absolute file path for file %s due to  %s", err, predictFilename)
	}
	x_predict, _ := operation.ReadFile(absFilePathPredict)
	l.Predict(x_predict)
}
