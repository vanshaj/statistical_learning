package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func printMat() {
	a := mat.NewDense(3, 1, []float64{
		3,
		4,
		5,
	})
	b := mat.NewDense(3, 1, []float64{
		5,
		6,
		7,
	})

	//Add used to add two martices
	var c mat.Dense
	c.Add(a, b)

	fc := mat.Formatted(&c, mat.Prefix(" "), mat.Squeeze())
	fmt.Printf("c = \n%v\n", fc)

	// Dims used to get Dimensions
	rows, cols := c.Dims()
	fmt.Printf("Dimensions of c are %d*%d\n", rows, cols)

}

func calSqrt() {
	// Calculate sqrt of a matrix
	rows := 2
	cols := 2
	squares := mat.NewDense(rows, cols, []float64{
		4, 16,
		9, 25,
	})
	roots := mat.NewDense(2, 2, []float64{
		0, 0,
		0, 0,
	})

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			actValue := squares.At(i, j)
			srtValue := math.Sqrt(actValue)
			roots.Set(i, j, srtValue)
		}
	}
	fc := mat.Formatted(roots, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("c = \n%v\n", fc)

}
