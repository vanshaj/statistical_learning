package main

import (
	"fmt"

	"github.com/vanshaj/cs229/pkg/neuralnet"
	"gonum.org/v1/gonum/mat"
)

func main() {
	//prediction.PredictPoisson()
	// neuralnet.ForwardPassExample()
	withDenseLayers()
}

func withDenseLayers() {
	h1 := neuralnet.NewHidderLayer(3, 4)
	inputF := []float64{
		1, 2, 3, 2.5,
		2, 5, -1, 2.0,
		-1.5, 2.7, 3.3, -0.8,
	}
	outputL1 := h1.CalculateForwardPassOutput(inputF, 3)
	h2 := neuralnet.NewHidderLayer(3, 3)
	outputL2 := h2.CalculateForwardPassOutputMat(outputL1)
	fmt.Printf("%v\n", mat.Formatted(outputL2, mat.Prefix("  ")))
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
