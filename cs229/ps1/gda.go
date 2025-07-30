package ps1

import (
	"log"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Gaussian Discriminant Analysis
type BinaryGda struct {
	phi   float64
	mean0 float64
	mean1 float64
	sigma *mat.Dense
}

// We have to calculate 4 parameters
// Phi, Mean_0, Mean_1, Covariance Matrix
func (g *BinaryGda) calculatePhi(y *mat.Dense) {
	numSamples, _ := y.Dims()
	numerator := 0.0
	for i := range numSamples {
		numerator = numerator + y.At(i, 0)
	}
	g.phi = numerator / float64(numSamples)
}

func (g *BinaryGda) calculateMean0(x, y *mat.Dense) {
	numSamples, _ := x.Dims()
	numerator := 0.0
	denominator := 0
	for i := range numSamples {
		val := 0
		if y.At(i, 0) == 0.0 {
			val = 1
		}
		denominator = denominator + val
	}
	for i := range numSamples {
		if y.At(i, 0) == 0.0 {
			numerator = numerator + x.At(i, 0)
		}
	}
	g.mean0 = numerator / float64(denominator)
}

func (g *BinaryGda) calculateMean1(x, y *mat.Dense) {
	numSamples, _ := x.Dims()
	numerator := 0.0
	denominator := 0
	for i := range numSamples {
		val := 0
		if y.At(i, 0) == 1.0 {
			val = 1
		}
		denominator = denominator + val
	}
	for i := range numSamples {
		if y.At(i, 0) == 1.0 {
			numerator = numerator + x.At(i, 0)
		}
	}
	g.mean1 = numerator / float64(denominator)
}

func (g *BinaryGda) calculateCovMatrix(x *mat.Dense) {
	numSamples, numFeatures := x.Dims()
	centeredVector := mat.NewDense(numSamples, numFeatures, nil)
	mu_0 := g.mean0
	mu_1 := g.mean1
	for i := range numSamples {
		for j := range numFeatures {
			if j == 0 {
				new_val := x.At(i, j) - mu_0
				centeredVector.Set(i, j, new_val)
			} else {
				new_val := x.At(i, j) - mu_1
				centeredVector.Set(i, j, new_val)
			}
		}
	}
	transposeXM := centeredVector.T()
	covMat := mat.NewDense(numFeatures, numFeatures, nil)
	covMat.Mul(transposeXM, centeredVector)
	for i := range numFeatures {
		for j := range numFeatures {
			covMat.Set(i, j, covMat.At(i, j)/float64(numSamples-1))
		}
	}
	g.sigma = covMat
}

func (g *BinaryGda) Fit(x, y *mat.Dense) {
	g.calculatePhi(y)
	g.calculateMean0(x, y)
	g.calculateMean1(x, y)
	g.calculateCovMatrix(x)
}

func (g *BinaryGda) calculatePDF(mean float64, x, covMat *mat.Dense, numFeatures int) float64 {
	numMulFactor := math.Pow(2*math.Pi, float64(numFeatures/2))
	denominator := numMulFactor * math.Pow(mat.Det(covMat), 0.5)
	expFactor := mat.NewDense(1, 1, nil)

	numSamples, _ := x.Dims()
	xCentered := mat.NewDense(numSamples, 1, nil)
	for i := range numSamples {
		xCentered.Set(i, 0, x.At(i, 0)-mean)
	}
	xCenteredT := xCentered.T()
	covMatI := mat.NewDense(numSamples, numSamples, nil)
	covMatI.Inverse(covMat)
	firstMul := mat.NewDense(1, numSamples, nil)
	firstMul.Mul(xCenteredT, covMatI)
	expFactor.Mul(firstMul, xCentered)

	exponentPower := (-0.5 * expFactor.At(0, 0))
	numerator := math.Pow(math.E, exponentPower)

	pdf := (numerator / denominator)
	return pdf
}

func (g *BinaryGda) calculateProb0(x *mat.Dense) float64 {
	_, numFeatures := x.Dims()
	pdf := g.calculatePDF(g.mean0, x, g.sigma, numFeatures)
	return (1 - g.phi) * pdf
}

func (g *BinaryGda) calculateProb1(x *mat.Dense) float64 {
	_, numFeatures := x.Dims()
	pdf := g.calculatePDF(g.mean1, x, g.sigma, numFeatures)
	return g.phi * pdf
}

func (g *BinaryGda) Predict(x *mat.Dense) {
	prob0 := g.calculateProb0(x)
	prob1 := g.calculateProb1(x)
	if prob0 > prob1 {
		log.Println("0.0")
	} else {
		log.Println("1.0")
	}
}
