# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, etensor, dog, vfilter, image

class Vis(pygiv.ClassViewObj):
    """
    Vis does DoG filtering on images
    """

    def __init__(self):
        super(Vis, self).__init__()
        self.ClipToFit = True
        self.SetTags("ClipToFit", 'desc:"if true, and input image is larger than target image size, central region is clipped out as the input -- otherwise image is sized to target size"')
        self.DoG = dog.Filter()
        self.SetTags("DoG", 'desc:"LGN DoG filter parameters"')
        self.Geom = vfilter.Geom()
        self.SetTags("Geom", 'inactive:"+" view:"inline" desc:"geometry of input, output"')
        self.ImgSize = image.Point()
        self.SetTags("ImgSize", 'desc:"target image size to use -- images will be rescaled to this size"')
        self.DoGTsr = etensor.Float32()
        self.SetTags("DoGTsr", 'view:"no-inline" desc:"DoG filter tensor"')
        self.Img = image.Image()
        self.SetTags("Img", 'view:"-" desc:"current input image"')
        self.ImgTsr = etensor.Float32()
        self.SetTags("ImgTsr", 'view:"no-inline" desc:"input image as tensor"')
        self.OutTsr = etensor.Float32()
        self.SetTags("OutTsr", 'view:"no-inline" desc:"DoG filter output tensor"')

    def Defaults(vi):
        vi.ClipToFit = True
        vi.DoG.Defaults()
        sz = 16
        spc = 2
        vi.DoG.SetSize(sz, spc)
        # note: first arg is border -- we are relying on Geom
        # to set border to .5 * filter size
        # any further border sizes on same image need to add Geom.FiltRt!
        vi.Geom.Set(image.Point(0, 0), image.Point(spc, spc), image.Point(sz, sz))
        vi.ImgSize = image.Point(24, 24)
        vi.Geom.SetSize(vi.ImgSize.Add(vi.Geom.Border.Mul(2)))
        vi.DoG.ToTensor(vi.DoGTsr)
        # vi.ImgTsr.SetMetaData("image", "+")
        vi.ImgTsr.SetMetaData("grid-fill", "1")

    def SetImage(vi, img):
        """
        SetImage sets current image for processing
        """
        vi.Img = img
        insz = vi.Geom.In
        ibd = img.Bounds()
        isz = ibd.Size()
        if vi.ClipToFit and isz.X > insz.X and isz.Y > insz.Y:
            st = isz.Sub(insz).Div(2).Add(ibd.Min)
            ed = st.Add(insz)
            vi.Img = image.RGBA(img).SubImage(image.Rectangle(Min= st, Max= ed))
            vfilter.RGBToGrey(vi.Img, vi.ImgTsr, 0, False)
        else:
            if isz != vi.ImgSize:
                vi.Img = transform.Resize(vi.Img, vi.ImgSize.X, vi.ImgSize.Y, transform.Linear)
                vfilter.RGBToGrey(vi.Img, vi.ImgTsr, vi.Geom.FiltRt.X, False) # pad for filt, bot zero
                vfilter.WrapPad(vi.ImgTsr, vi.Geom.FiltRt.X)

    def LGNDoG(vi):
        """
        LGNDoG runs DoG filtering on input image
        must have valid Img in place to start.
        """
        flt = vi.DoG.FilterTensor(vi.DoGTsr, dog.Net)
        vfilter.Conv1(vi.Geom, flt, vi.ImgTsr, vi.OutTsr, vi.DoG.Gain)

        vfilter.TensorLogNorm32(vi.OutTsr, 0) # 0 = renorm all, 1 = renorm within each on / off separately

    def Filter(vi, img):
        """
        Filter is overall method to run filters on given image
        """
        vi.SetImage(img)
        vi.LGNDoG()

