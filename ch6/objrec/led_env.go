// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"golang.org/x/exp/rand"
)

// LEDEnv generates images of old-school "LED" style "letters" composed of a set of horizontal
// and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
// Renders using SVG.
type LEDEnv struct {
	Nm     string  `desc:"name of this environment"`
	Dsc    string  `desc:"description of this environment"`
	Draw   LEDraw  `desc:"draws LEDs onto image"`
	MinLED int     `min:"0" max:"19" desc:"minimum LED number to draw (0-19)"`
	MaxLED int     `min:"0" max:"19" desc:"maximum LED number to draw (0-19)"`
	CurLED int     `inactive:"+" desc:"current LED number that was drawn"`
	PrvLED int     `inactive:"+" desc:"previous LED number that was drawn"`
	Run    env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch  env.Ctr `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial  env.Ctr `view:"inline" desc:"trial is the step counter within epoch"`
}

func (le *LEDEnv) Name() string { return le.Nm }
func (le *LEDEnv) Desc() string { return le.Dsc }

func (le *LEDEnv) Validate() error {
	return nil
}

func (le *LEDEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (le *LEDEnv) States() env.Elements {
	sz := le.Draw.ImgSize
	els := env.Elements{
		{"OrigImage", []int{sz.Y, sz.X}, []string{"Y", "X"}},
		{"Image", []int{sz.Y, sz.X}, []string{"Y", "X"}},
	}
	return els
}

func (le *LEDEnv) State(element string) etensor.Tensor {
	// switch element {
	// case "NNext":
	// 	return &le.NNext
	// case "NextStates":
	// 	return &le.NextStates
	// case "NextLabels":
	// 	return &le.NextLabels
	// }
	// return nil
	return nil
}

func (le *LEDEnv) Actions() env.Elements {
	return nil
}

func (le *LEDEnv) Init(run int) {
	le.Draw.Init()
	le.Run.Scale = env.Run
	le.Epoch.Scale = env.Epoch
	le.Trial.Scale = env.Trial
	le.Run.Init()
	le.Epoch.Init()
	le.Trial.Init()
	le.Run.Cur = run
	le.Trial.Max = 0
	le.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

func (le *LEDEnv) Step() bool {
	le.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	le.Trial.Incr()
	le.DrawLED()
	// then distort
	return true
}

func (le *LEDEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (le *LEDEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return le.Run.Query()
	case env.Epoch:
		return le.Epoch.Query()
	case env.Trial:
		return le.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*LEDEnv)(nil)

// DrawLED picks a new random LED and draws it
func (le *LEDEnv) DrawLED() {
	rng := 1 + le.MaxLED - le.MinLED
	led := le.MinLED + rand.Intn(rng)
	le.Draw.Clear()
	le.Draw.DrawLED(led)
	le.PrvLED = le.CurLED
	le.CurLED = led
}
