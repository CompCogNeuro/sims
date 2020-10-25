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
	"math/rand"

	"github.com/anthonynsimon/bild/clone"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/emer/vision/vxform"
	"github.com/goki/gi/gi"
)

// ImgEnv presents images from a list of image files, using V1 simple and complex filtering.
// images are just selected at random each trial -- nothing fancy here.
type ImgEnv struct {
	Nm         string          `desc:"name of this environment"`
	Dsc        string          `desc:"description of this environment"`
	ImageFiles []string        `desc:"paths to images"`
	Images     []*image.RGBA   `desc:"images (preload for speed)"`
	ImageIdx   env.CurPrvInt   `desc:"current image index"`
	Vis        Vis             `desc:"visual processing params"`
	XFormRand  vxform.Rand     `desc:"random transform parameters"`
	XForm      vxform.XForm    `desc:"current -- prev transforms"`
	Run        env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial      env.Ctr         `view:"inline" desc:"trial is the step counter within epoch"`
	OrigImg    etensor.Float32 `desc:"original image prior to random transforms"`
}

func (ev *ImgEnv) Name() string { return ev.Nm }
func (ev *ImgEnv) Desc() string { return ev.Dsc }

func (ev *ImgEnv) Validate() error {
	return nil
}

func (ev *ImgEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (ev *ImgEnv) States() env.Elements {
	isz := ev.Vis.ImgSize
	sz := ev.Vis.OutTsr.Shapes()
	nms := ev.Vis.OutTsr.DimNames()
	els := env.Elements{
		{"Image", []int{isz.Y, isz.X}, []string{"Y", "X"}},
		{"LGN", sz, nms},
		{"LGNon", sz[1:], nms[1:]},
		{"LGNoff", sz[1:], nms[1:]},
	}
	return els
}

func (ev *ImgEnv) State(element string) etensor.Tensor {
	switch element {
	case "Image":
		return &ev.Vis.ImgTsr
	case "LGN":
		return &ev.Vis.OutTsr
	case "LGNon":
		return ev.Vis.OutTsr.SubSpace([]int{0})
	case "LGNoff":
		return ev.Vis.OutTsr.SubSpace([]int{1})
	}
	return nil
}

func (ev *ImgEnv) Actions() env.Elements {
	return nil
}

func (ev *ImgEnv) Defaults() {
	ev.Vis.Defaults()
	ev.XFormRand.TransX.Set(0, 0) // translation happens in random chunk selection
	ev.XFormRand.TransY.Set(0, 0)
	ev.XFormRand.Scale.Set(0.5, 1)
	ev.XFormRand.Rot.Set(-90, 90)
}

func (ev *ImgEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

func (ev *ImgEnv) Step() bool {
	ev.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ev.Trial.Incr() { // if true, hit max, reset to 0
		ev.Epoch.Incr()
	}
	ev.PickRndImage()
	ev.FilterImg()
	// debug only:
	// img := ev.Images[ev.ImageIdx.Cur]
	// vfilter.RGBToGrey(img, &ev.OrigImg, 0, false) // pad for filt, bot zero
	return true
}

// DoImage processes specified image number
func (ev *ImgEnv) DoImage(imgNo int) {
	ev.ImageIdx.Set(imgNo)
	ev.FilterImg()
}

func (ev *ImgEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *ImgEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
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
var _ env.Env = (*ImgEnv)(nil)

// String returns the string rep of the LED env state
func (ev *ImgEnv) String() string {
	cfn := ev.ImageFiles[ev.ImageIdx.Cur]
	return fmt.Sprintf("Obj: %s, %s", cfn, ev.XForm.String())
}

// PickRndImage picks an image at random
func (ev *ImgEnv) PickRndImage() {
	nimg := len(ev.Images)
	ev.ImageIdx.Set(rand.Intn(nimg))
}

// FilterImg filters the image using new random xforms
func (ev *ImgEnv) FilterImg() {
	ev.XFormRand.Gen(&ev.XForm)
	oimg := ev.Images[ev.ImageIdx.Cur]
	// following logic first extracts a sub-image of 2x the ultimate filtered size of image
	// from original image, which greatly speeds up the xform processes, relative to working
	// on entire 800x600 original image
	insz := ev.Vis.Geom.In.Mul(2) // target size * 2
	ibd := oimg.Bounds()
	isz := ibd.Size()
	irng := isz.Sub(insz)
	var st image.Point
	st.X = rand.Intn(irng.X)
	st.Y = rand.Intn(irng.Y)
	ed := st.Add(insz)
	simg := oimg.SubImage(image.Rectangle{Min: st, Max: ed})
	img := ev.XForm.Image(simg)
	ev.Vis.Filter(img)
}

// OpenImages opens all the images
func (ev *ImgEnv) OpenImages() error {
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
		ev.Images[i] = gi.ImageToRGBA(img)
	}
	return lsterr
}

// OpenImagesAsset opens all the images as assets
func (ev *ImgEnv) OpenImagesAsset() error {
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
