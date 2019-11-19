// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"

	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/etable/etensor"
	"github.com/emer/vision/dog"
	"github.com/emer/vision/vfilter"
	"github.com/goki/ki/kit"
)

// Vis does DoG filtering on images
type Vis struct {
	ClipToFit bool            `desc:"if true, and input image is larger than target image size, central region is clipped out as the input -- otherwise image is sized to target size"`
	DoG       dog.Filter      `desc:"LGN DoG filter parameters"`
	Geom      vfilter.Geom    `inactive:"+" view:"inline" desc:"geometry of input, output"`
	ImgSize   image.Point     `desc:"target image size to use -- images will be rescaled to this size"`
	DoGTsr    etensor.Float32 `view:"no-inline" desc:"DoG filter tensor"`
	Img       image.Image     `view:"-" desc:"current input image"`
	ImgTsr    etensor.Float32 `view:"no-inline" desc:"input image as tensor"`
	OutTsr    etensor.Float32 `view:"no-inline" desc:"DoG filter output tensor"`
}

var KiT_Vis = kit.Types.AddType(&Vis{}, nil)

func (vi *Vis) Defaults() {
	vi.ClipToFit = true
	vi.DoG.Defaults()
	sz := 16
	spc := 2
	vi.DoG.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.Geom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.ImgSize = image.Point{24, 24}
	vi.Geom.SetSize(vi.ImgSize.Add(vi.Geom.Border.Mul(2)))
	vi.DoG.ToTensor(&vi.DoGTsr)
	// vi.ImgTsr.SetMetaData("image", "+")
	vi.ImgTsr.SetMetaData("grid-fill", "1")
}

// SetImage sets current image for processing
func (vi *Vis) SetImage(img image.Image) {
	vi.Img = img
	insz := vi.Geom.In
	ibd := img.Bounds()
	isz := ibd.Size()
	if vi.ClipToFit && isz.X > insz.X && isz.Y > insz.Y {
		st := isz.Sub(insz).Div(2).Add(ibd.Min)
		ed := st.Add(insz)
		vi.Img = img.(*image.RGBA).SubImage(image.Rectangle{Min: st, Max: ed})
		vfilter.RGBToGrey(vi.Img, &vi.ImgTsr, 0, false) // pad for filt, bot zero
	} else {
		if isz != vi.ImgSize {
			vi.Img = transform.Resize(vi.Img, vi.ImgSize.X, vi.ImgSize.Y, transform.Linear)
			vfilter.RGBToGrey(vi.Img, &vi.ImgTsr, vi.Geom.FiltRt.X, false) // pad for filt, bot zero
			vfilter.WrapPad(&vi.ImgTsr, vi.Geom.FiltRt.X)
		}
	}
}

// LGNDoG runs DoG filtering on input image
// must have valid Img in place to start.
func (vi *Vis) LGNDoG() {
	flt := vi.DoG.FilterTensor(&vi.DoGTsr, dog.Net)
	vfilter.Conv1(&vi.Geom, flt, &vi.ImgTsr, &vi.OutTsr, vi.DoG.Gain)
	// log norm is generally good it seems for dogs
	vfilter.TensorLogNorm32(&vi.OutTsr, 0) // 0 = renorm all, 1 = renorm within each on / off separately
}

// Filter is overall method to run filters on given image
func (vi *Vis) Filter(img image.Image) {
	vi.SetImage(img)
	vi.LGNDoG()
}
