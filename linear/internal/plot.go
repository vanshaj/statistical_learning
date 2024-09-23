package internal

import (
	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/db"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

func PlotLR(x_query, y_query string, l *LinearCoeff) error {
	y_values, err := db.GetValues(y_query)
	if err != nil {
		return err
	}
	x_values, err := db.GetValues(x_query)
	if err != nil {
		return err
	}
	p := plot.New()
	p.Title.Text = "Medv Vs Lstat"
	p.X.Label.Text = "Lstat"
	p.Y.Label.Text = "Medv"

	p.Add(plotter.NewGrid())
	pts := make(plotter.XYs, len(x_values))
	ptsLinear := make(plotter.XYs, len(x_values))
	for i, val := range x_values {
		pts[i].X = val
		ptsLinear[i].X = val
		ptsLinear[i].Y = l.Intercept + (l.Slope * val)
	}
	for i, val := range y_values {
		pts[i].Y = val
	}
	s, err := plotter.NewScatter(pts)
	if err != nil {
		log.Error().Msgf("Unable to create scatter plot due to %s\n", err)
		return err
	}
	p.Add(s)
	linearLine, err := plotter.NewLine(ptsLinear)
	if err != nil {
		log.Error().Msgf("Unable to create line due to %s\n", err)
		return err
	}
	p.Add(linearLine)
	if err = p.Save(400, 200, "MedvVsLstat.png"); err != nil {
		log.Error().Msgf("Unable to save scatter plot due to %s\n", err)
		return err
	}
	return nil
}
