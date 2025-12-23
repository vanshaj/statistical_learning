package main

import (
	"fmt"

	"github.com/vanshaj/cs229/pkg/neuralnet"
	"gonum.org/v1/gonum/mat"
)

func main() {
	//prediction.PredictPoisson()
	forwardPass()
}

func simpleNeuron() {
	input := []float64{
		1, 2, 3, 2.5,
	}
	weights := [][]float64{
		{0.2, 0.8, -0.5, 1},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87}}
	bias := []float64{2, 3, 0.5}
	// Number of neurons will be similar to the number of biases as each neuron will have it's bias
	neurons := make([]neuralnet.Neuron, len(bias))
	for i := range bias {
		neuron := neuralnet.Neuron{
			Weights: weights[i],
			Bias:    bias[i],
		}
		neurons[i] = neuron
	}
	nn := neuralnet.NeuralNetworkLayer{
		Neurons: neurons,
	}
	outputs := nn.CalculateLayerOutput(input)
	for i := range len(outputs) {
		fmt.Println(outputs[i])
	}

}

func forwardPass() {
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
	hl := neuralnet.HiddenLayer{
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
	hl2 := neuralnet.HiddenLayer{
		Weights:          weightsF2,
		Bias:             biasF2,
		NeuronCount:      3,
		WeightInputCount: 3,
	}
	finalOutputMat := hl2.CalculateForwardPassOutputMat(outputMat)
	fmt.Printf("%v\n", mat.Formatted(finalOutputMat, mat.Prefix("  ")))

}
