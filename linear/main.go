package main

import (
	"github.com/rs/zerolog/log"
	"github.com/vanshaj/statistical_learning/linear/app"
)

func main() {
	if err := app.Run(); err != nil {
		log.Panic().Msg(err.Error())
	}
}
