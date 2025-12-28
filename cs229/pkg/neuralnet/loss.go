package neuralnet

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

/*
Here we are calculating cross entropy loss
Assumption here is the y_actual matrix is one hot encoded
y_predicted will be m*n matrix m-> number of input data set , n-> number of neurons
y_actual will be m*n matrix(only with 0/1) m-> number of input data set, n-> number of neuron / number of classes for classification
*/
func CrossEntropyLoss(y_predicted, y_actual *mat.Dense) float64 {
	/*
		We don't want our predicted value to reach absolute 0 or absolute 1 so we will set limit before applying loss
	*/
	y_predicted.Apply(func(i, j int, v float64) float64 {
		maxVal := float64(1) - float64(1e-7)
		minVal := float64(1e-7)
		if v > maxVal {
			return maxVal
		} else if v < minVal {
			return minVal
		} else {
			return v
		}
	}, y_predicted)
	/*
	 In order to calculate loss we will perform y_predicted[i][j] * y_actual[i][j]
	 This will ensure only those item remains in y_predicted that have been identified corrector
	*/
	m, n := y_predicted.Dims()
	y_correct_pred_mat := mat.NewDense(m, n, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			y_correct_pred_mat.Set(i, j, y_predicted.At(i, j)*y_actual.At(i, j))
		}
	}

	// Now we will apply negative log on each item
	y_correct_pred_mat.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return (0 - math.Log(v))
		} else {
			return 0
		}
	}, y_correct_pred_mat)

	// Now we will sum all the elements
	return mat.Sum(y_correct_pred_mat)
}
