// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"log"

	"github.com/anthonynsimon/bild/clone"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
)

// SemEnv presents paragraphs of text, loaded from file(s)
// This assumes files have all been pre-filtered so only relevant words are present.
type SemEnv struct {
	Nm         string         `desc:"name of this environment"`
	Dsc        string         `desc:"description of this environment"`
	Sequential bool           `desc:"if true, go sequentially through paragraphs -- else permuted"`
	TextFiles  []string       `desc:"paths to text files"`
	Words      []string       `desc:"list of words"`
	Paras      [][]string     `desc:"paragraphs"`
	CurPara    etable.Float32 `desc:"current paragraph rendered as localist word units"`
	ParaIdx    env.CurPrvInt  `desc:"current paragraph index"`
	Run        env.Ctr        `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr        `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial      env.Ctr        `view:"inline" desc:"trial is the step counter within epoch"`
}

func (ev *SemEnv) Name() string { return ev.Nm }
func (ev *SemEnv) Desc() string { return ev.Dsc }

func (ev *SemEnv) Validate() error {
	return nil
}

func (ev *SemEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (ev *SemEnv) States() env.Elements {
	sz := ev.CurPara.Shapes()
	nms := ev.CurPara.DimNames()
	els := env.Elements{
		{"Input", sz, nms},
	}
	return els
}

func (ev *SemEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.CurPara
	}
	return nil
}

func (ev *SemEnv) Actions() env.Elements {
	return nil
}

func (ev *SemEnv) Defaults() {
}

func (ev *SemEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

func (ev *SemEnv) Step() bool {
	ev.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ev.Trial.Incr() { // if true, hit max, reset to 0
		ev.Epoch.Incr()
	}
	// ev.PickRndImage()
	// ev.FilterImg()
	return true
}

func (ev *SemEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *SemEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*SemEnv)(nil)

// String returns the string rep of the LED env state
func (ev *SemEnv) String() string {
	cfn := ev.ImageFiles[ev.ImageIdx.Cur]
	return fmt.Sprintf("Obj: %s, %s", cfn, ev.XForm.String())
}

// OpenImages opens all the images
func (ev *SemEnv) OpenImages() error {
	nimg := len(ev.ImageFiles)
	if len(ev.Images) != nimg {
		ev.Images = make([]*image.RGBA, nimg)
	}
	var lsterr error
	for i, fn := range ev.ImageFiles {
		img, err := gi.OpenImage(fn)
		if err != nil {
			log.Println(err)
			lsterr = err
			continue
		}
		if rg, ok := img.(*image.RGBA); ok {
			ev.Images[i] = rg
		} else {
			ev.Images[i] = clone.AsRGBA(img)
		}
	}
	return lsterr
}

// OpenImagesAsset opens all the images as assets
func (ev *SemEnv) OpenImagesAsset() error {
	nimg := len(ev.ImageFiles)
	if len(ev.Images) != nimg {
		ev.Images = make([]*image.RGBA, nimg)
	}
	var lsterr error
	for i, fn := range ev.ImageFiles {
		ab, err := Asset(fn) // embedded in executable
		if err != nil {
			log.Println(err)
			lsterr = err
			continue
		}
		img, err := jpeg.Decode(bytes.NewBuffer(ab))
		if err != nil {
			log.Println(err)
			lsterr = err
			continue
		}
		if rg, ok := img.(*image.RGBA); ok {
			ev.Images[i] = rg
		} else {
			ev.Images[i] = clone.AsRGBA(img)
		}
	}
	return lsterr
}
