// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/emer/vision/vfilter"
	"github.com/emer/vision/vxform"
)

// ImgEnv presents images from a list of image files, using V1 simple and complex filtering.
type ImgEnv struct {
	Nm        string          `desc:"name of this environment"`
	Dsc       string          `desc:"description of this environment"`
	Vis       Vis             `desc:"visual processing params"`
	XFormRand vxform.Rand     `desc:"random transform parameters"`
	XForm     vxform.XForm    `desc:"current -- prev transforms"`
	Run       env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch     env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial     env.Ctr         `view:"inline" desc:"trial is the step counter within epoch"`
	OrigImg   etensor.Float32 `desc:"original image prior to random transforms"`
}

func (le *ImgEnv) Name() string { return le.Nm }
func (le *ImgEnv) Desc() string { return le.Dsc }

func (le *ImgEnv) Validate() error {
	return nil
}

func (le *ImgEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (le *ImgEnv) States() env.Elements {
	isz := le.Draw.ImgSize
	sz := le.Vis.V1AllTsr.Shapes()
	nms := le.Vis.V1AllTsr.DimNames()
	els := env.Elements{
		{"Image", []int{isz.Y, isz.X}, []string{"Y", "X"}},
		{"V1", sz, nms},
	}
	return els
}

func (le *ImgEnv) State(element string) etensor.Tensor {
	switch element {
	case "Image":
		vfilter.RGBToGrey(le.Draw.Image, &le.OrigImg, 0, false) // pad for filt, bot zero
		return &le.OrigImg
	case "V1":
		return &le.Vis.V1AllTsr
	}
	return nil
}

func (le *ImgEnv) Actions() env.Elements {
	return nil
}

func (le *ImgEnv) Defaults() {
	le.Draw.Defaults()
	le.Vis.Defaults()
	le.XFormRand.TransX.Set(-0.125, 0.125)
	le.XFormRand.TransY.Set(-0.125, 0.125)
	le.XFormRand.Scale.Set(0.7, 1)
	le.XFormRand.Rot.Set(-3.6, 3.6)
}

func (le *ImgEnv) Init(run int) {
	le.Draw.Init()
	le.Run.Scale = env.Run
	le.Epoch.Scale = env.Epoch
	le.Trial.Scale = env.Trial
	le.Run.Init()
	le.Epoch.Init()
	le.Trial.Init()
	le.Run.Cur = run
	le.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

func (le *ImgEnv) Step() bool {
	le.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if le.Trial.Incr() { // if true, hit max, reset to 0
		le.Epoch.Incr()
	}
	le.FilterImg()
	// debug only:
	// vfilter.RGBToGrey(le.Draw.Image, &le.OrigImg, 0, false) // pad for filt, bot zero
	return true
}

// DoObject renders specific object (LED number)
// func (le *ImgEnv) DoObject(objno int) {
// 	le.DrawLED(objno)
// 	le.FilterImg()
// }

func (le *ImgEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (le *ImgEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
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
var _ env.Env = (*ImgEnv)(nil)

// String returns the string rep of the LED env state
func (le *ImgEnv) String() string {
	return fmt.Sprintf("Obj: %02d, %s", le.CurLED, le.XForm.String())
}

// DrawRndLED picks a new random LED and draws it
func (le *ImgEnv) DrawRndLED() {
	rng := 1 + le.MaxLED - le.MinLED
	led := le.MinLED + rand.Intn(rng)
	le.DrawLED(led)
}

// DrawLED draw specified LED
func (le *ImgEnv) DrawLED(led int) {
	le.Draw.Clear()
	le.Draw.DrawLED(led)
	le.PrvLED = le.CurLED
	le.CurLED = led
	le.SetOutput(le.CurLED)
}

// FilterImg filters the image from LED
func (le *ImgEnv) FilterImg() {
	le.XFormRand.Gen(&le.XForm)
	img := le.XForm.Image(le.Draw.Image)
	le.Vis.Filter(img)
}
