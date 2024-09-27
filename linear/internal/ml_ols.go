package internal

import (
	"github.com/rs/zerolog/log"
	"gonum.org/v1/gonum/mat"
)

type MultipleCoeff struct{ Coefficients []float64 }

func NewMultipleCoeff() *MultipleCoeff {
	return &MultipleCoeff{Coefficients: make([]float64, 0)}
}

// x is all the values of predictors, y is all the value of response
// n is the number of response values and p is the number of predictors
func (m *MultipleCoeff) CalculateCoefficient(x, y []float64, n, p int) {
	/*
		We will create 2 matrices
		Y = n*1 matrix
		X = n*p matrix
		Coefficient Matrix(p*1) = (X^TX)^-1X^TY
	*/
	Y_mat := mat.NewDense(n, 1, y)
	X_mat := mat.NewDense(n, p, x)
	X_out := mat.NewDense(p, p, make([]float64, p*p))
	var X_t mat.Matrix = X_mat.T()
	t_r, t_c := X_t.Dims()
	log.Debug().Msgf("Dimensions of X Transpose : %d, %d", t_r, t_c)
	log.Debug().Msgf("Dimensions of X: %d, %d", n, p)
	t_r, t_c = X_out.Dims()
	log.Debug().Msgf("Dimensions of X output: %d, %d", t_r, t_c)
	X_out.Mul(X_t, X_mat)
	X_out.Inverse(X_out)
	X_out_2 := mat.NewDense(p, n, make([]float64, p*n))
	X_out_2.Mul(X_out, X_t)
	X_out_3 := mat.NewDense(p, 1, make([]float64, p*1))
	X_out_3.Mul(X_out_2, Y_mat)

	for i := 0; i < p; i++ {
		m.Coefficients = append(m.Coefficients, X_out_3.At(i, 0))
	}
}
