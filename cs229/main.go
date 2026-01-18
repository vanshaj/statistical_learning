package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/vanshaj/cs229/pkg/neuralnet"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Seed for reproducibility
	rand.Seed(time.Now().UnixNano())

	// 1. DATA GENERATION
	// Create 300 samples: 2 features (x, y coordinates)
	nSamples := 300
	nFeatures := 2
	nClasses := 3

	X := mat.NewDense(nSamples, nFeatures, nil)
	yTrue := mat.NewDense(nSamples, nClasses, nil)

	// Create 3 clusters of data points
	for i := 0; i < nSamples; i++ {
		class := i % nClasses
		// Center points around (0.2, 0.2), (0.8, 0.8), and (0.2, 0.8)
		var cx, cy float64
		switch class {
		case 0:
			cx, cy = 0.2, 0.2
		case 1:
			cx, cy = 0.8, 0.8
		case 2:
			cx, cy = 0.2, 0.8
		}
		X.Set(i, 0, cx+rand.NormFloat64()*0.1)
		X.Set(i, 1, cy+rand.NormFloat64()*0.1)
		yTrue.Set(i, class, 1.0)
	}

	// 2. NETWORK ARCHITECTURE
	// Layer 1: 2 inputs -> 16 hidden neurons
	layer1 := neuralnet.NewHiddenLayer(16, nFeatures)
	// Layer 2: 16 hidden neurons -> 3 output classes
	layer2 := neuralnet.NewHiddenLayer(nClasses, 16)

	learningRate := 0.1
	epochs := 2001

	fmt.Printf("Training on %d samples...\n", nSamples)
	fmt.Println("-------------------------------------------")

	// 3. TRAINING LOOP
	for epoch := 0; epoch < epochs; epoch++ {

		// --- FORWARD PASS ---
		// Input -> Layer 1
		z1 := layer1.CalculateForwardPassOutput(X)
		layer1.ActivationReLU(z1) // z1 is now the activated output

		// Layer 1 Output -> Layer 2
		z2 := layer2.CalculateForwardPassOutput(z1)
		layer2.ActivationSoftmax(z2) // z2 is now probabilities

		// --- LOSS ---
		loss := layer2.CategoricalCrossEntropy(z2, yTrue)

		// --- BACKWARD PASS ---
		// 1. Initial Gradient: (yPred - yTrue) / nSamples
		dZ2 := mat.NewDense(nSamples, nClasses, nil)
		dZ2.Sub(z2, yTrue)
		dZ2.Scale(1.0/float64(nSamples), dZ2)

		// 2. Backprop through Layer 2
		dInputs2 := layer2.BackwardPassOutput(dZ2)

		// 3. Backprop through ReLU (using z1 which was the input to the activation)
		dReLU := layer1.ActivationReLUBackward(dInputs2, z1)

		// 4. Backprop through Layer 1
		layer1.BackwardPassOutput(dReLU)

		// --- OPTIMIZATION (Update Weights) ---
		layer1.UpdateParameters(learningRate)
		layer2.UpdateParameters(learningRate)

		// Logging
		if epoch%200 == 0 {
			fmt.Printf("Epoch %4d | Loss: %.6f\n", epoch, loss)
		}
	}

	// 4. SIMPLE INFERENCE TEST
	fmt.Println("-------------------------------------------")
	fmt.Println("Test Inference (Point near Cluster 1 [0.2, 0.2]):")
	testPoint := mat.NewDense(1, 2, []float64{0.15, 0.15})

	// Forward pass for the test point
	p1 := layer1.CalculateForwardPassOutput(testPoint)
	layer1.ActivationReLU(p1)
	p2 := layer2.CalculateForwardPassOutput(p1)
	layer2.ActivationSoftmax(p2)

	fmt.Printf("Class Probabilities: %v\n", mat.Formatted(p2))
}
