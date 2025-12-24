package neuralnet

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type HiddenLayer struct {
	Weights          []float64
	Bias             []float64
	NeuronCount      int // Number of neurons in this layer
	WeightInputCount int // number of weights for neuron per input as we have
}

func ForwardPassExample() {
	// We have 3 set of inputs and each input has 4 values
	inputF := []float64{
		1, 2, 3, 2.5,
		2, 5, -1, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	}
	/* Assuming we have 3 neurons
	Each has 4 weights for each value of 1 input set
	*/
	weightsF := []float64{
		0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87,
	}
	/*
		We have 3 neurons and each has one bias
	*/
	biasF := []float64{2, 3, 0.5}
	hl := HiddenLayer{
		Weights:          weightsF,
		Bias:             biasF,
		NeuronCount:      3, // Number of neurons in this layer
		WeightInputCount: 4, // weight for neuron per input as we have 4 inputs in each set
	}
	outputMat := hl.CalculateForwardPassOutput(inputF, 3) // output will be 3 * 3 matrix as we have 3 neurons
	// fmt.Printf("%v\n", mat.Formatted(outputMat, mat.Prefix("  ")))

	/*
		For new layers let's assume we have 3 more neurons
		And has 3 weights for each value of 1 ouput
	*/
	weightsF2 := []float64{
		0.1, -0.14, 0.5,
		-0.5, 0.12, -0.33,
		-0.44, 0.73, -0.13,
	}
	biasF2 := []float64{-1, 2, -0.5}
	hl2 := HiddenLayer{
		Weights:          weightsF2,
		Bias:             biasF2,
		NeuronCount:      3,
		WeightInputCount: 3,
	}
	finalOutputMat := hl2.CalculateForwardPassOutputMat(outputMat)
	fmt.Printf("%v\n", mat.Formatted(finalOutputMat, mat.Prefix("  ")))

}

func NewHidderLayer(n_Neurons, n_Input int) *HiddenLayer {
	weights := make([]float64, n_Input*n_Neurons)
	bias := make([]float64, n_Neurons)
	// Will create weights and bias automatically
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

func (hl *HiddenLayer) ActivationSoftmax(inputMat *mat.Dense) {
	r, c := inputMat.Dims()
	// First we will get max from each row
	maxMat := mat.NewDense(r, c, nil)
}

// First we will apply ReLU and then we will go to forward Pass
func (hl *HiddenLayer) ActivationReLU(inputMat *mat.Dense) {
	// Apply will apply your func on each input of the Dense (matrix)
	inputMat.Apply(func(r, c int, v float64) float64 {
		if v > 0 {
			return v
		} else {
			return 0
		}
	}, inputMat)
}

func (hl *HiddenLayer) CalculateForwardPassOutputMat(inputMat *mat.Dense) *mat.Dense {
	inputRowCount, _ := inputMat.Dims()
	weightsMat := mat.NewDense(hl.NeuronCount, hl.WeightInputCount, hl.Weights) // Weight matrix of x * n size

	// Apply ReluActivation
	hl.ActivationReLU(inputMat)

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
	inputMat := mat.NewDense(inputCount, hl.WeightInputCount, input) // Input matrix of m * n size
	return hl.CalculateForwardPassOutputMat(inputMat)
}
