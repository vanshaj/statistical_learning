package neuralnet

import (
	"fmt"
	"math"
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
	/*
		There will be X rows of inputs but each row will have n_Input columns
		For each input among n_Input each neuron will have a weight
		and there are n_Neurons
		so the size of weight matrix is n_Input * n_Neurons
	*/
	weights := make([]float64, n_Input*n_Neurons)

	// Every neuron will have 1 bias
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

// This will be applied at the end output
func (hl *HiddenLayer) ActivationSoftmax(inputMat *mat.Dense) {
	r, _ := inputMat.Dims()
	// First we will get max from each row
	ithRowMaxMat := mat.NewVecDense(r, nil)
	for i := range r {
		ithRow := inputMat.RowView(i)
		ithRowMaxMat.SetVec(i, mat.Max(ithRow))
	}
	// Before we apply exponential to each value we will subtract each value from max of that row to protect exp to become too large
	inputMat.Apply(func(i, j int, v float64) float64 {
		// i will be row and j will be column
		var output float64
		// First let's shift left the value
		v = ithRowMaxMat.At(i, 0) - v
		// Now lets apply exponential
		output = math.Exp(v)
		return output
	}, inputMat)
	// Now create a vector that will contain sum of each values of a row after we calculate exponent
	ithRowSumMat := mat.NewVecDense(r, nil)
	for i := range r {
		ithRow := inputMat.RowView(i)
		ithRowSumMat.SetVec(i, mat.Sum(ithRow))
	}
	// Now divide each value from the sum of each values in that row
	inputMat.Apply(func(i, j int, v float64) float64 {
		var output float64
		output = v / ithRowSumMat.At(i, 0)
		return output
	}, inputMat)
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
