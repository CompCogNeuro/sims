// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objrec

import (
	"image"

	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorcore"
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/v1vision/fffb"
	"github.com/emer/v1vision/gabor"
	"github.com/emer/v1vision/kwta"
	"github.com/emer/v1vision/v1complex"
	"github.com/emer/v1vision/vfilter"
)

// Vis encapsulates specific visual processing pipeline for V1 filtering
type Vis struct { //types:add

	// V1 simple gabor filter parameters
	V1sGabor gabor.Filter

	// geometry of input, output for V1 simple-cell processing
	V1sGeom vfilter.Geom `edit:"-" display:"inline"`

	// neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code
	V1sNeighInhib kwta.NeighInhib

	// kwta parameters for V1s
	V1sKWTA kwta.KWTA

	// target image size to use -- images will be rescaled to this size
	ImgSize image.Point

	// V1 simple gabor filter tensor
	V1sGaborTsr tensor.Float32 `display:"no-inline"`

	// input image as tensor
	ImgTsr tensor.Float32 `display:"no-inline"`

	// current input image
	Img image.Image `display:"-"`

	// V1 simple gabor filter output tensor
	V1sTsr tensor.Float32 `display:"no-inline"`

	// V1 simple extra Gi from neighbor inhibition tensor
	V1sExtGiTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, kwta output tensor
	V1sKwtaTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of V1sKwta tensor
	V1sPoolTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, un-max-pooled 2x2 of V1sPool tensor
	V1sUnPoolTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, angle-only features tensor
	V1sAngOnlyTsr tensor.Float32 `display:"no-inline"`

	// V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor
	V1sAngPoolTsr tensor.Float32 `display:"no-inline"`

	// V1 complex length sum filter output tensor
	V1cLenSumTsr tensor.Float32 `display:"no-inline"`

	// V1 complex end stop filter output tensor
	V1cEndStopTsr tensor.Float32 `display:"no-inline"`

	// Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total
	V1AllTsr tensor.Float32 `display:"no-inline"`

	// inhibition values for V1s KWTA
	V1sInhibs fffb.Inhibs `display:"no-inline"`
}

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
	tensorcore.AddGridStylerTo(&vi.ImgTsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
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
	ny := vi.V1sPoolTsr.DimSize(0)
	nx := vi.V1sPoolTsr.DimSize(1)
	nang := vi.V1sPoolTsr.DimSize(3)
	nrows := 5
	vi.V1AllTsr.SetShapeSizes(ny, nx, nrows, nang)
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
