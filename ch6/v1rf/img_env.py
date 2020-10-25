# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, env, etensor, vxform, image, gi, rand

from dogfilter import Vis

class ImgEnv(pygiv.ClassViewObj):
    """
    ImgEnv presents images from a list of image files, using V1 simple and complex filtering.
    images are just selected at random each trial -- nothing fancy here.
    """

    def __init__(self):
        super(ImgEnv, self).__init__()
        self.Nm = str()
        self.SetTags("Nm", 'desc:"name of this environment"')
        self.Dsc = str()
        self.SetTags("Dsc", 'desc:"description of this environment"')
        self.ImageFiles = []
        self.SetTags("ImageFiles", 'desc:"paths to images"')
        self.Images = []
        self.SetTags("Images", 'desc:"images (preload for speed)"')
        self.ImageIdx = env.CurPrvInt()
        self.SetTags("ImageIdx", 'desc:"current image index"')
        self.Vis = Vis()
        self.SetTags("Vis", 'desc:"visual processing params"')
        self.XFormRand = vxform.Rand()
        self.SetTags("XFormRand", 'desc:"random transform parameters"')
        self.XForm = vxform.XForm()
        self.SetTags("XForm", 'desc:"current -- prev transforms"')
        self.Run = env.Ctr()
        self.SetTags("Run", 'view:"inline" desc:"current run of model as provided during Init"')
        self.Epoch = env.Ctr()
        self.SetTags("Epoch", 'view:"inline" desc:"number of times through Seq.Max number of sequences"')
        self.Trial = env.Ctr()
        self.SetTags("Trial", 'view:"inline" desc:"trial is the step counter within epoch"')
        self.OrigImg = etensor.Float32()
        self.SetTags("OrigImg", 'desc:"original image prior to random transforms"')

    def Name(ev):
        return ev.Nm

    def Desc(ev):
        return ev.Dsc

    def Validate(ev):
        return 0

    def State(ev, element):
        if element == "Image":
            return ev.Vis.ImgTsr
        if element == "LGN":
            return ev.Vis.OutTsr
        if element == "LGNon":
            return ev.Vis.OutTsr.SubSpace(go.Slice_int([0]))
        if element == "LGNoff":
            return ev.Vis.OutTsr.SubSpace(go.Slice_int([1]))
        return go.nil

    def Defaults(ev):
        ev.Vis.Defaults()
        ev.XFormRand.TransX.Set(0, 0) # translation happens in random chunk selection
        ev.XFormRand.TransY.Set(0, 0)
        ev.XFormRand.Scale.Set(0.5, 1)
        ev.XFormRand.Rot.Set(-90, 90)

    def Init(ev, run):
        ev.Run.Scale = env.Run
        ev.Epoch.Scale = env.Epoch
        ev.Trial.Scale = env.Trial
        ev.Run.Init()
        ev.Epoch.Init()
        ev.Trial.Init()
        ev.Run.Cur = run
        ev.Trial.Cur = -1 # init state -- key so that first Step() = 0

    def Step(ev):
        ev.Epoch.Same()     # good idea to just reset all non-inner-most counters at start
        if ev.Trial.Incr(): # if true, hit max, reset to 0
            ev.Epoch.Incr()
        ev.PickRndImage()
        ev.FilterImg()
        # debug only:
        # img := ev.Images[ev.ImageIdx.Cur]
        # vfilter.RGBToGrey(img, &ev.OrigImg, 0, false) // pad for filt, bot zero
        return True

    def DoImage(ev, imgNo):
        """
        DoImage processes specified image number
        """
        ev.ImageIdx.Set(imgNo)
        ev.FilterImg()

    def Action(ev, element, input):
        pass

    def CounterCur(ev, scale):
        if scale == env.Run:
            return ev.Run.Cur
        if scale == env.Epoch:
            return ev.Epoch.Cur
        if scale == env.Trial:
            return ev.Trial.Cur
        return -1

    def CounterPrv(ev, scale):
        if scale == env.Run:
            return ev.Run.Prv
        if scale == env.Epoch:
            return ev.Epoch.Prv
        if scale == env.Trial:
            return ev.Trial.Prv
        return -1
        
    def CounterChg(ev, scale):
        if scale == env.Run:
            return ev.Run.Chg
        if scale == env.Epoch:
            return ev.Epoch.Chg
        if scale == env.Trial:
            return ev.Trial.Chg
        return -1

    def String(ev):
        """
        String returns the string rep of the LED env state
        """
        cfn = ev.ImageFiles[ev.ImageIdx.Cur]
        return "Obj: %s, %s" % (cfn, ev.XForm.String())

    def PickRndImage(ev):
        """
        PickRndImage picks an image at random
        """
        nimg = len(ev.Images)
        ev.ImageIdx.Set(rand.Intn(nimg))

    def FilterImg(ev):
        """
        FilterImg filters the image using new random xforms
        """
        ev.XFormRand.Gen(ev.XForm)
        oimg = ev.Images[ev.ImageIdx.Cur]

        insz = ev.Vis.Geom.In.Mul(2)
        ibd = oimg.Bounds()
        isz = ibd.Size()
        irng = isz.Sub(insz)
        st = image.Point()
        st.X = rand.Intn(irng.X)
        st.Y = rand.Intn(irng.Y)
        ed = st.Add(insz)
        simg = oimg.SubImage(image.Rectangle(Min= st, Max= ed))
        img = ev.XForm.Image(simg)
        ev.Vis.Filter(img)

    def OpenImages(ev):
        """
        OpenImages opens all the images
        """
        nimg = len(ev.ImageFiles)
        if len(ev.Images) != nimg:
            ev.Images = []
        for fn in ev.ImageFiles:
            img = gi.ImageToRGBA(gi.OpenImage(fn))
            ev.Images.append(img)

