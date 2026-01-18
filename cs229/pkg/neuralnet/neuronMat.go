package neuralnet

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type HiddenLayer struct {
	Weights          *mat.Dense
	Bias             *mat.Dense
	Input            *mat.Dense
	WeightDerivative *mat.Dense
	BiasDerivative   *mat.Dense
	InputDerivative  *mat.Dense
}

// NewHiddenLayer initializes a layer with weights (Neurons x Features) and bias (1 x Neurons)
func NewHiddenLayer(nNeurons, nInputFeatures int) *HiddenLayer {
	// Initialize weights with small random values
	// Shape: (nNeurons x nInputFeatures)
	weightsData := make([]float64, nNeurons*nInputFeatures)
	for i := range weightsData {
		weightsData[i] = 0.01 * rand.NormFloat64()
	}
	weights := mat.NewDense(nNeurons, nInputFeatures, weightsData)

	// Initialize bias as a row vector (1 x nNeurons)
	biasData := make([]float64, nNeurons)
	// Bias usually starts at 0
	bias := mat.NewDense(1, nNeurons, biasData)

	return &HiddenLayer{
		Weights: weights,
		Bias:    bias,
	}
}

// CalculateForwardPassOutput computes: Output = (Input * Weights^T) + Bias
func (hl *HiddenLayer) CalculateForwardPassOutput(input *mat.Dense) *mat.Dense {
	hl.Input = input // Store for backprop
	nSamples, _ := input.Dims()
	nNeurons, _ := hl.Weights.Dims()

	// 1. Dot Product: (Samples x Features) * (Features x Neurons) = (Samples x Neurons)
	outputMat := mat.NewDense(nSamples, nNeurons, nil)
	outputMat.Mul(input, hl.Weights.T())

	// 2. Add Bias: Add the row vector hl.Bias to every row of outputMat
	for i := 0; i < nSamples; i++ {
		row := outputMat.RawRowView(i)
		for j := 0; j < nNeurons; j++ {
			row[j] += hl.Bias.At(0, j)
		}
	}

	return outputMat
}

// ActivationReLU applies the Rectified Linear Unit element-wise
func (hl *HiddenLayer) ActivationReLU(inputMat *mat.Dense) {
	inputMat.Apply(func(_, _ int, v float64) float64 {
		return math.Max(0, v)
	}, inputMat)
}

// ActivationReLUBackward computes gradient for ReLU
func (hl *HiddenLayer) ActivationReLUBackward(dValues, layerInput *mat.Dense) *mat.Dense {
	r, c := dValues.Dims()
	grad := mat.NewDense(r, c, nil)
	grad.Apply(func(i, j int, v float64) float64 {
		// If input was > 0, gradient is passed through
		if layerInput.At(i, j) > 0 {
			return dValues.At(i, j)
		}
		return 0
	}, grad)
	return grad
}

// ActivationSoftmax normalizes rows into probability distributions (Stable version)
func (hl *HiddenLayer) ActivationSoftmax(inputMat *mat.Dense) {
	r, c := inputMat.Dims()
	for i := 0; i < r; i++ {
		row := inputMat.RawRowView(i)

		// Find max for numerical stability
		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}

		// Subtract max and exponentiate
		var sum float64
		for j := 0; j < c; j++ {
			row[j] = math.Exp(row[j] - maxVal)
			sum += row[j]
		}

		// Normalize to get probabilities
		for j := 0; j < c; j++ {
			row[j] /= sum
		}
	}
}

// CategoricalCrossEntropy calculates average negative log likelihood
func (hl *HiddenLayer) CategoricalCrossEntropy(yPred, yTrue *mat.Dense) float64 {
	nSamples, nClasses := yPred.Dims()
	var totalLoss float64

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nClasses; j++ {
			if yTrue.At(i, j) > 0 {
				val := yPred.At(i, j)
				// Clip to prevent log(0)
				val = math.Max(1e-7, math.Min(1.0-1e-7, val))
				totalLoss += -math.Log(val)
			}
		}
	}
	return totalLoss / float64(nSamples)
}

// BackwardPassOutput calculates gradients for Weights, Bias, and Input
func (hl *HiddenLayer) BackwardPassOutput(dValues *mat.Dense) *mat.Dense {
	nSamples, _ := dValues.Dims()

	// 1. Weight Derivative: dW = dValues^T * Input
	// (Neurons x Samples) * (Samples x Features) = (Neurons x Features)
	wdR, wdC := hl.Weights.Dims()
	hl.WeightDerivative = mat.NewDense(wdR, wdC, nil)
	hl.WeightDerivative.Mul(dValues.T(), hl.Input)

	// 2. Bias Derivative: Sum dValues columns
	hl.BiasDerivative = sumAxis0(dValues)

	// 3. Input Derivative: dInput = dValues * Weights
	// (Samples x Neurons) * (Neurons x Features) = (Samples x Features)
	_, nFeatures := hl.Weights.Dims()
	hl.InputDerivative = mat.NewDense(nSamples, nFeatures, nil)
	hl.InputDerivative.Mul(dValues, hl.Weights)

	return hl.InputDerivative
}

// sumAxis0 is the Gonum equivalent of np.sum(array, axis=0, keepdims=True)
func sumAxis0(dValues *mat.Dense) *mat.Dense {
	r, c := dValues.Dims()
	ones := mat.NewDense(1, r, nil)
	for i := 0; i < r; i++ {
		ones.Set(0, i, 1.0)
	}
	result := mat.NewDense(1, c, nil)
	result.Mul(ones, dValues)
	return result
}

func (hl *HiddenLayer) UpdateParameters(learningRate float64) {
	// 1. Update Weights: W = W - (learningRate * dW)
	// We use hl.WeightDerivative.Scale to multiply by learningRate
	wR, wC := hl.Weights.Dims()
	scaledDW := mat.NewDense(wR, wC, nil)
	scaledDW.Scale(learningRate, hl.WeightDerivative)
	hl.Weights.Sub(hl.Weights, scaledDW)

	// 2. Update Bias: B = B - (learningRate * dB)
	bR, bC := hl.Bias.Dims()
	scaledDB := mat.NewDense(bR, bC, nil)
	scaledDB.Scale(learningRate, hl.BiasDerivative)
	hl.Bias.Sub(hl.Bias, scaledDB)
}
