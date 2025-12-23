package neuralnet

import (
	"gonum.org/v1/gonum/mat"
)

type Neuron struct {
	Weights []float64
	Bias    float64
}

type NeuralNetworkLayer struct {
	Neurons []Neuron
}

func (n *Neuron) calculateNeuronOutput(input []float64) float64 {
	// var output float64
	// for i := range input {
	// 	output = output + n.Weights[i]*input[i]
	// }
	// output = output + n.Bias
	// return output

	inputVec := mat.NewVecDense(len(input), input)           // represents a vector n*1
	weightsVec := mat.NewVecDense(len(n.Weights), n.Weights) // represents a vector n*1
	dotProduct := mat.Dot(inputVec, weightsVec)              // dot product
	return dotProduct + n.Bias
}

func (nn *NeuralNetworkLayer) CalculateLayerOutput(input []float64) []float64 {
	nnOutput := make([]float64, len(nn.Neurons))
	for i := range len(nn.Neurons) {
		nOutput := nn.Neurons[i].calculateNeuronOutput(input)
		nnOutput[i] = nOutput
	}
	return nnOutput
}
