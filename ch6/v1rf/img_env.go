// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"log"
	"math/rand"

	"github.com/anthonynsimon/bild/clone"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/emer/vision/vfilter"
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

func (ie *ImgEnv) Name() string { return ie.Nm }
func (ie *ImgEnv) Desc() string { return ie.Dsc }

func (ie *ImgEnv) Validate() error {
	return nil
}

func (ie *ImgEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (ie *ImgEnv) States() env.Elements {
	isz := ie.Vis.ImgSize
	sz := ie.Vis.OutTsr.Shapes()
	nms := ie.Vis.OutTsr.DimNames()
	els := env.Elements{
		{"Image", []int{isz.Y, isz.X}, []string{"Y", "X"}},
		{"LGN", sz, nms},
		{"LGNon", sz[1:], nms[1:]},
		{"LGNoff", sz[1:], nms[1:]},
	}
	return els
}

func (ie *ImgEnv) State(element string) etensor.Tensor {
	switch element {
	case "Image":
		return &ie.Vis.ImgTsr
	case "LGN":
		return &ie.Vis.OutTsr
	case "LGNon":
		return ie.Vis.OutTsr.SubSpace(2, []int{0})
	case "LGNoff":
		return ie.Vis.OutTsr.SubSpace(2, []int{1})
	}
	return nil
}

func (ie *ImgEnv) Actions() env.Elements {
	return nil
}

func (ie *ImgEnv) Defaults() {
	ie.Vis.Defaults()
	ie.XFormRand.TransX.Set(0, 0) // translation happens in random chunk selection
	ie.XFormRand.TransY.Set(0, 0)
	ie.XFormRand.Scale.Set(0.5, 1)
	ie.XFormRand.Rot.Set(-90, 90)
}

func (ie *ImgEnv) Init(run int) {
	ie.Run.Scale = env.Run
	ie.Epoch.Scale = env.Epoch
	ie.Trial.Scale = env.Trial
	ie.Run.Init()
	ie.Epoch.Init()
	ie.Trial.Init()
	ie.Run.Cur = run
	ie.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

func (ie *ImgEnv) Step() bool {
	ie.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ie.Trial.Incr() { // if true, hit max, reset to 0
		ie.Epoch.Incr()
	}
	ie.PickRndImage()
	ie.FilterImg()
	// debug only:
	img := ie.Images[ie.ImageIdx.Cur]
	vfilter.RGBToGrey(img, &ie.OrigImg, 0, false) // pad for filt, bot zero
	return true
}

// DoImage processes specified image number
func (ie *ImgEnv) DoImage(imgNo int) {
	ie.ImageIdx.Set(imgNo)
	ie.FilterImg()
}

func (ie *ImgEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ie *ImgEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ie.Run.Query()
	case env.Epoch:
		return ie.Epoch.Query()
	case env.Trial:
		return ie.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*ImgEnv)(nil)

// String returns the string rep of the LED env state
func (ie *ImgEnv) String() string {
	cfn := ie.ImageFiles[ie.ImageIdx.Cur]
	return fmt.Sprintf("Obj: %s, %s", cfn, ie.XForm.String())
}

// PickRndImage picks an image at random
func (ie *ImgEnv) PickRndImage() {
	nimg := len(ie.Images)
	ie.ImageIdx.Set(rand.Intn(nimg))
}

// FilterImg filters the image using new random xforms
func (ie *ImgEnv) FilterImg() {
	ie.XFormRand.Gen(&ie.XForm)
	oimg := ie.Images[ie.ImageIdx.Cur]
	// following logic first extracts a sub-image of 2x the ultimate filtered size of image
	// from original image, which greatly speeds up the xform processes, relative to working
	// on entire 800x600 original image
	insz := ie.Vis.Geom.In.Mul(2) // target size * 2
	ibd := oimg.Bounds()
	isz := ibd.Size()
	irng := isz.Sub(insz)
	var st image.Point
	st.X = rand.Intn(irng.X)
	st.Y = rand.Intn(irng.Y)
	ed := st.Add(insz)
	simg := oimg.SubImage(image.Rectangle{Min: st, Max: ed})
	img := ie.XForm.Image(simg)
	ie.Vis.Filter(img)
}

// OpenImages opens all the images
func (ie *ImgEnv) OpenImages() error {
	nimg := len(ie.ImageFiles)
	if len(ie.Images) != nimg {
		ie.Images = make([]*image.RGBA, nimg)
	}
	var lsterr error
	for i, fn := range ie.ImageFiles {
		img, err := gi.OpenImage(fn)
		if err != nil {
			log.Println(err)
			lsterr = err
		}
		if rg, ok := img.(*image.RGBA); ok {
			ie.Images[i] = rg
		} else {
			ie.Images[i] = clone.AsRGBA(img)
		}
	}
	return lsterr
}
