// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"io/fs"
	"log"
	"math/rand"

	"cogentcore.org/core/base/iox/imagex"
	"github.com/anthonynsimon/bild/clone"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/etensor/tensor"
	"github.com/emer/vision/v2/vxform"
)

// ImgEnv presents images from a list of image files, using V1 simple and complex filtering.
// images are just selected at random each trial -- nothing fancy here.
type ImgEnv struct {
	// name of this environment
	Name string
	// paths to images
	ImageFiles []string
	// images (preload for speed)
	Images []*image.RGBA
	// current image index
	ImageIndex env.CurPrvInt
	// visual processing params
	Vis Vis
	// random transform parameters
	XFormRand vxform.Rand
	// current -- prev transforms
	XForm vxform.XForm
	// current run of model as provided during Init
	Trial env.Counter `view:"inline"`
	// original image prior to random transforms
	OrigImg tensor.Float32
}

func (ev *ImgEnv) Label() string { return ev.Name }

func (ev *ImgEnv) State(element string) tensor.Tensor {
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

func (ev *ImgEnv) Defaults() {
	ev.Vis.Defaults()
	ev.XFormRand.TransX.Set(0, 0) // translation happens in random chunk selection
	ev.XFormRand.TransY.Set(0, 0)
	ev.XFormRand.Scale.Set(0.5, 1)
	ev.XFormRand.Rot.Set(-90, 90)
}

func (ev *ImgEnv) Init(run int) {
	ev.Trial.Scale = etime.Trial
	ev.Trial.Init()
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

func (ev *ImgEnv) Step() bool {
	ev.Trial.Incr()
	ev.PickRndImage()
	ev.FilterImg()
	// debug only:
	// img := ev.Images[ev.ImageIndex.Cur]
	// vfilter.RGBToGrey(img, &ev.OrigImg, 0, false) // pad for filt, bot zero
	return true
}

// DoImage processes specified image number
func (ev *ImgEnv) DoImage(imgNo int) {
	ev.ImageIndex.Set(imgNo)
	ev.FilterImg()
}

func (ev *ImgEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*ImgEnv)(nil)

// String returns the string rep of the LED env state
func (ev *ImgEnv) String() string {
	cfn := ev.ImageFiles[ev.ImageIndex.Cur]
	return fmt.Sprintf("Obj: %s, %s", cfn, ev.XForm.String())
}

// PickRndImage picks an image at random
func (ev *ImgEnv) PickRndImage() {
	nimg := len(ev.Images)
	ev.ImageIndex.Set(rand.Intn(nimg))
}

// FilterImg filters the image using new random xforms
func (ev *ImgEnv) FilterImg() {
	ev.XFormRand.Gen(&ev.XForm)
	oimg := ev.Images[ev.ImageIndex.Cur]
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
		img, _, err := imagex.Open(fn)
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

// OpenImagesFS opens all the images as assets
func (ev *ImgEnv) OpenImagesFS(fsys fs.FS) error {
	nimg := len(ev.ImageFiles)
	if len(ev.Images) != nimg {
		ev.Images = make([]*image.RGBA, nimg)
	}
	var lsterr error
	for i, fn := range ev.ImageFiles {
		img, _, err := imagex.OpenFS(fsys, fn)
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
