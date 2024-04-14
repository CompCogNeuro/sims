# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, gi, pygiv, etensor, gabor, image, fffb, kwta, v1complex, vfilter


class Vis(pyviews.ClassViewObj):
    """
    Vis encapsulates specific visual processing pipeline for V1 filtering
    """

    def __init__(self):
        super(Vis, self).__init__()
        self.V1sGabor = gabor.Filter()
        self.SetTags("V1sGabor", 'desc:"V1 simple gabor filter parameters"')
        self.V1sGeom = vfilter.Geom()
        self.SetTags(
            "V1sGeom",
            'inactive:"+" view:"inline" desc:"geometry of input, output for V1 simple-cell processing"',
        )
        self.V1sNeighInhib = kwta.NeighInhib()
        self.SetTags(
            "V1sNeighInhib",
            'desc:"neighborhood inhibition for V1s -- each unit gets inhibition from same feature in nearest orthogonal neighbors -- reduces redundancy of feature code"',
        )
        self.V1sKWTA = kwta.KWTA()
        self.SetTags("V1sKWTA", 'desc:"kwta parameters for V1s"')
        self.ImgSize = image.Point()
        self.SetTags(
            "ImgSize",
            'desc:"target image size to use -- images will be rescaled to this size"',
        )
        self.V1sGaborTsr = etensor.Float32()
        self.SetTags(
            "V1sGaborTsr", 'view:"no-inline" desc:"V1 simple gabor filter tensor"'
        )
        self.ImgTsr = etensor.Float32()
        self.SetTags("ImgTsr", 'view:"no-inline" desc:"input image as tensor"')
        self.Img = image.Image()
        self.SetTags("Img", 'view:"-" desc:"current input image"')
        self.V1sTsr = etensor.Float32()
        self.SetTags(
            "V1sTsr", 'view:"no-inline" desc:"V1 simple gabor filter output tensor"'
        )
        self.V1sExtGiTsr = etensor.Float32()
        self.SetTags(
            "V1sExtGiTsr",
            'view:"no-inline" desc:"V1 simple extra Gi from neighbor inhibition tensor"',
        )
        self.V1sKwtaTsr = etensor.Float32()
        self.SetTags(
            "V1sKwtaTsr",
            'view:"no-inline" desc:"V1 simple gabor filter output, kwta output tensor"',
        )
        self.V1sPoolTsr = etensor.Float32()
        self.SetTags(
            "V1sPoolTsr",
            'view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of V1sKwta tensor"',
        )
        self.V1sUnPoolTsr = etensor.Float32()
        self.SetTags(
            "V1sUnPoolTsr",
            'view:"no-inline" desc:"V1 simple gabor filter output, un-max-pooled 2x2 of V1sPool tensor"',
        )
        self.V1sAngOnlyTsr = etensor.Float32()
        self.SetTags(
            "V1sAngOnlyTsr",
            'view:"no-inline" desc:"V1 simple gabor filter output, angle-only features tensor"',
        )
        self.V1sAngPoolTsr = etensor.Float32()
        self.SetTags(
            "V1sAngPoolTsr",
            'view:"no-inline" desc:"V1 simple gabor filter output, max-pooled 2x2 of AngOnly tensor"',
        )
        self.V1cLenSumTsr = etensor.Float32()
        self.SetTags(
            "V1cLenSumTsr",
            'view:"no-inline" desc:"V1 complex length sum filter output tensor"',
        )
        self.V1cEndStopTsr = etensor.Float32()
        self.SetTags(
            "V1cEndStopTsr",
            'view:"no-inline" desc:"V1 complex end stop filter output tensor"',
        )
        self.V1AllTsr = etensor.Float32()
        self.SetTags(
            "V1AllTsr",
            'view:"no-inline" desc:"Combined V1 output tensor with V1s simple as first two rows, then length sum, then end stops = 5 rows total"',
        )
        self.V1sInhibs = fffb.Inhibs()
        self.SetTags(
            "V1sInhibs", 'view:"no-inline" desc:"inhibition values for V1s KWTA"'
        )

    def Defaults(vi):
        vi.V1sGabor.Defaults()
        sz = 6  # V1mF16 typically = 12, no border, spc = 4 -- using 1/2 that here
        spc = 2
        vi.V1sGabor.SetSize(sz, spc)
        # note: first arg is border -- we are relying on Geom
        # to set border to .5 * filter size
        # any further border sizes on same image need to add Geom.FiltRt!
        vi.V1sGeom.Set(image.Point(0, 0), image.Point(spc, spc), image.Point(sz, sz))
        vi.V1sNeighInhib.Defaults()
        vi.V1sKWTA.Defaults()
        vi.ImgSize = image.Point(40, 40)
        vi.V1sGabor.ToTensor(vi.V1sGaborTsr)
        # vi.ImgTsr.SetMetaData("image", "+")
        vi.ImgTsr.SetMetaData("grid-fill", "1")

    def SetImage(vi, img):
        """
        SetImage sets current image for processing
        """
        vi.Img = img
        isz = vi.Img.Bounds().Size()
        if isz != vi.ImgSize:
            vi.Img = core.ImageResize(vi.Img, vi.ImgSize.X, vi.ImgSize.Y)
        vfilter.RGBToGrey(vi.Img, vi.ImgTsr, vi.V1sGeom.FiltRt.X, False)
        vfilter.WrapPad(vi.ImgTsr, vi.V1sGeom.FiltRt.X)

    def V1Simple(vi):
        """
        V1Simple runs V1Simple Gabor filtering on input image
        must have valid Img in place to start.
        Runs kwta and pool steps after gabor filter.
        """
        vfilter.Conv(vi.V1sGeom, vi.V1sGaborTsr, vi.ImgTsr, vi.V1sTsr, vi.V1sGabor.Gain)
        if vi.V1sNeighInhib.On:
            vi.V1sNeighInhib.Inhib4(vi.V1sTsr, vi.V1sExtGiTsr)
        else:
            vi.V1sExtGiTsr.SetZeros()
        if vi.V1sKWTA.On:
            vi.V1sKWTA.KWTAPool(vi.V1sTsr, vi.V1sKwtaTsr, vi.V1sInhibs, vi.V1sExtGiTsr)
        else:
            vi.V1sKwtaTsr.CopyFrom(vi.V1sTsr)

    def V1Complex(vi):
        """
        it computes Angle-only, max-pooled version of V1Simple inputs.
        """
        vfilter.MaxPool(
            image.Point(2, 2), image.Point(2, 2), vi.V1sKwtaTsr, vi.V1sPoolTsr
        )
        vfilter.MaxReduceFilterY(vi.V1sKwtaTsr, vi.V1sAngOnlyTsr)
        vfilter.MaxPool(
            image.Point(2, 2), image.Point(2, 2), vi.V1sAngOnlyTsr, vi.V1sAngPoolTsr
        )
        v1complex.LenSum4(vi.V1sAngPoolTsr, vi.V1cLenSumTsr)
        v1complex.EndStop4(vi.V1sAngPoolTsr, vi.V1cLenSumTsr, vi.V1cEndStopTsr)

    def V1All(vi):
        """
        V1All aggregates all the relevant simple and complex features
        into the V1AllTsr which is used for input to a network
        """
        ny = vi.V1sPoolTsr.Dim(0)
        nx = vi.V1sPoolTsr.Dim(1)
        nang = vi.V1sPoolTsr.Dim(3)
        nrows = 5
        oshp = go.Slice_int([ny, nx, nrows, nang])
        if not etensor.EqualInts(oshp, vi.V1AllTsr.Shp):
            vi.V1AllTsr.SetShape(
                oshp, go.nil, go.Slice_string(["Y", "X", "Polarity", "Angle"])
            )

        vfilter.FeatAgg(go.Slice_int([0]), 0, vi.V1cLenSumTsr, vi.V1AllTsr)

        vfilter.FeatAgg(go.Slice_int([0, 1]), 1, vi.V1cEndStopTsr, vi.V1AllTsr)

        vfilter.FeatAgg(go.Slice_int([0, 1]), 3, vi.V1sPoolTsr, vi.V1AllTsr)

    def Filter(vi, img):
        """
        Filter is overall method to run filters on given image
        """
        vi.SetImage(img)
        vi.V1Simple()
        vi.V1Complex()
        vi.V1All()
