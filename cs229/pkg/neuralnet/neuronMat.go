package neuralnet

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type HiddenLayer struct {
	Weights          []float64
	Bias             []float64
	NeuronCount      int // Number of neurons in this layer
	WeightInputCount int // number of weights for neuron per input as we have
}

func NewHidderLayer(n_Neurons, n_Input int) *HiddenLayer {
	weights := make([]float64, n_Input*n_Neurons)
	bias := make([]float64, n_Neurons)
	for i := range n_Input * n_Neurons {
		weights[i] = 0.01 * rand.Float64()
	}
	for i := range n_Neurons {
		bias[i] = 0.0
	}
	return &HiddenLayer{
		Weights:          weights,
		Bias:             bias,
		NeuronCount:      n_Neurons,
		WeightInputCount: n_Input,
	}
}

func (hl *HiddenLayer) CalculateForwardPassOutputMat(inputMat *mat.Dense) *mat.Dense {
	inputRowCount, _ := inputMat.Dims()
	weightsMat := mat.NewDense(hl.NeuronCount, hl.WeightInputCount, hl.Weights) // Weight matrix of x * n size
	biasExpand := make([]float64, hl.NeuronCount*len(hl.Bias))
	for i := range biasExpand {
		index := i % hl.NeuronCount
		biasExpand[i] = hl.Bias[index]
	}
	biasExpandMat := mat.NewDense(inputRowCount, hl.NeuronCount, biasExpand)

	// Transpose weight to make it n * x size
	weightsMatT := weightsMat.T()

	// Calculate output i.e m * x size
	outputMat := mat.NewDense(inputRowCount, hl.NeuronCount, nil)
	outputMat.Mul(inputMat, weightsMatT)
	outputMat.Add(outputMat, biasExpandMat)
	return outputMat
}

func (hl *HiddenLayer) CalculateForwardPassOutput(input []float64, inputCount int) *mat.Dense {
	inputMat := mat.NewDense(inputCount, hl.WeightInputCount, input)            // Input matrix of m * n size
	weightsMat := mat.NewDense(hl.NeuronCount, hl.WeightInputCount, hl.Weights) // Weight matrix of x * n size

	// Bias is of 1 * x but we want to later add it so we make it n*x
	// We duplicated elements of first row in second and then next till n
	biasExpand := make([]float64, hl.NeuronCount*len(hl.Bias))
	for i := range biasExpand {
		index := i % hl.NeuronCount
		biasExpand[i] = hl.Bias[index]
	}
	biasExpandMat := mat.NewDense(inputCount, hl.NeuronCount, biasExpand)

	// Transpose weight to make it n * x size
	weightsMatT := weightsMat.T()

	// Calculate output i.e m * x size
	outputMat := mat.NewDense(inputCount, hl.NeuronCount, nil)
	outputMat.Mul(inputMat, weightsMatT)
	outputMat.Add(outputMat, biasExpandMat)
	return outputMat
}
