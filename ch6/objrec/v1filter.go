// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"

	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/fffb"
	"github.com/emer/vision/gabor"
	"github.com/emer/vision/kwta"
	"github.com/emer/vision/v1complex"
	"github.com/emer/vision/vfilter"
	"github.com/goki/ki/kit"
)

// Vis encapsulates specific visual processing pipeline for V1 filtering
type Vis struct {
	V1sGabor      gabor.Filter    `desc:"V1 simple gabor filter parameters"`
	V1sGeom       vfilter.Geom    `inactive:"+" view:"inline" desc:"geometry of input, output for V1 simple-cell processing"`
	V1sNeighInhib kwta.NeighInhib `desc:"neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code"`
	V1sKWTA       kwta.KWTA       `desc:"kwta parameters for V1s"`
	ImgSize       image.Point     `desc:"target image size to use -- images will be rescaled to this size"`
	V1sGaborTsr   etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter tensor"`
	ImgTsr        etensor.Float32 `view:"no-inline" desc:"input image as tensor"`
	Img           image.Image     `view:"-" desc:"current input image"`
	V1sTsr        etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output tensor"`
	V1sExtGiTsr   etensor.Float32 `view:"no-inline" desc:"V1 simple extra Gi from neighbor inhibition tensor"`
	V1sKwtaTsr    etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, kwta output tensor"`
	V1sPoolTsr    etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of V1sKwta tensor"`
	V1sUnPoolTsr  etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, un-max-pooled 2x2 of V1sPool tensor"`
	V1sAngOnlyTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, angle-only features tensor"`
	V1sAngPoolTsr etensor.Float32 `view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor"`
	V1cLenSumTsr  etensor.Float32 `view:"no-inline" desc:"V1 complex length sum filter output tensor"`
	V1cEndStopTsr etensor.Float32 `view:"no-inline" desc:"V1 complex end stop filter output tensor"`
	V1AllTsr      etensor.Float32 `view:"no-inline" desc:"Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total"`
	V1sInhibs     fffb.Inhibs     `view:"no-inline" desc:"inhibition values for V1s KWTA"`
}

var KiT_Vis = kit.Types.AddType(&Vis{}, nil)

func (vi *Vis) Defaults() {
	vi.V1sGabor.Defaults()
	sz := 6 // V1mF16 typically = 12, no border, spc = 4 -- using 1/2 that here
	spc := 2
	vi.V1sGabor.SetSize(sz, spc)
	// note: first arg is border -- we are relying on Geom
	// to set border to .5 * filter size
	// any further border sizes on same image need to add Geom.FiltRt!
	vi.V1sGeom.Set(image.Point{0, 0}, image.Point{spc, spc}, image.Point{sz, sz})
	vi.V1sNeighInhib.Defaults()
	vi.V1sKWTA.Defaults()
	vi.ImgSize = image.Point{40, 40}
	vi.V1sGabor.ToTensor(&vi.V1sGaborTsr)
	// vi.ImgTsr.SetMetaData("image", "+")
	vi.ImgTsr.SetMetaData("grid-fill", "1")
}

// SetImage sets current image for processing
func (vi *Vis) SetImage(img image.Image) {
	vi.Img = img
	isz := vi.Img.Bounds().Size()
	if isz != vi.ImgSize {
		vi.Img = transform.Resize(vi.Img, vi.ImgSize.X, vi.ImgSize.Y, transform.Linear)
	}
	vfilter.RGBToGrey(vi.Img, &vi.ImgTsr, vi.V1sGeom.FiltRt.X, false) // pad for filt, bot zero
	vfilter.WrapPad(&vi.ImgTsr, vi.V1sGeom.FiltRt.X)
}

// V1Simple runs V1Simple Gabor filtering on input image
// must have valid Img in place to start.
// Runs kwta and pool steps after gabor filter.
func (vi *Vis) V1Simple() {
	vfilter.Conv(&vi.V1sGeom, &vi.V1sGaborTsr, &vi.ImgTsr, &vi.V1sTsr, vi.V1sGabor.Gain)
	if vi.V1sNeighInhib.On {
		vi.V1sNeighInhib.Inhib4(&vi.V1sTsr, &vi.V1sExtGiTsr)
	} else {
		vi.V1sExtGiTsr.SetZeros()
	}
	if vi.V1sKWTA.On {
		vi.V1sKWTA.KWTAPool(&vi.V1sTsr, &vi.V1sKwtaTsr, &vi.V1sInhibs, &vi.V1sExtGiTsr)
	} else {
		vi.V1sKwtaTsr.CopyFrom(&vi.V1sTsr)
	}
}

// it computes Angle-only, max-pooled version of V1Simple inputs.
func (vi *Vis) V1Complex() {
	vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sKwtaTsr, &vi.V1sPoolTsr)
	vfilter.MaxReduceFilterY(&vi.V1sKwtaTsr, &vi.V1sAngOnlyTsr)
	vfilter.MaxPool(image.Point{2, 2}, image.Point{2, 2}, &vi.V1sAngOnlyTsr, &vi.V1sAngPoolTsr)
	v1complex.LenSum4(&vi.V1sAngPoolTsr, &vi.V1cLenSumTsr)
	v1complex.EndStop4(&vi.V1sAngPoolTsr, &vi.V1cLenSumTsr, &vi.V1cEndStopTsr)
}

// V1All aggregates all the relevant simple and complex features
// into the V1AllTsr which is used for input to a network
func (vi *Vis) V1All() {
	ny := vi.V1sPoolTsr.Dim(0)
	nx := vi.V1sPoolTsr.Dim(1)
	nang := vi.V1sPoolTsr.Dim(3)
	nrows := 5
	oshp := []int{ny, nx, nrows, nang}
	if !etensor.EqualInts(oshp, vi.V1AllTsr.Shp) {
		vi.V1AllTsr.SetShape(oshp, nil, []string{"Y", "X", "Polarity", "Angle"})
	}
	// 1 length-sum
	vfilter.FeatAgg([]int{0}, 0, &vi.V1cLenSumTsr, &vi.V1AllTsr)
	// 2 end-stop
	vfilter.FeatAgg([]int{0, 1}, 1, &vi.V1cEndStopTsr, &vi.V1AllTsr)
	// 2 pooled simple cell
	vfilter.FeatAgg([]int{0, 1}, 3, &vi.V1sPoolTsr, &vi.V1AllTsr)
}

// Filter is overall method to run filters on given image
func (vi *Vis) Filter(img image.Image) {
	vi.SetImage(img)
	vi.V1Simple()
	vi.V1Complex()
	vi.V1All()
}
