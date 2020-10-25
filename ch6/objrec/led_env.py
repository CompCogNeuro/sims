# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

package main

import "fmt"
"math/rand"

"github.com/emer/emergent/env"
"github.com/emer/etable/etensor"
"github.com/emer/vision/vfilter"
"github.com/emer/vision/vxform"

class LEDEnv(pygiv.ClassViewObj):
    """
    LEDEnv generates images of old-school "LED" style "letters" composed of a set of horizontal
    and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
    Renders using SVG.
    """

    def __init__(self):
        super(Sim, self).__init__()
        self.Nm = str()
        self.SetTags("Nm", 'desc:"name of this environment"')
        self.Dsc = str()
        self.SetTags("Dsc", 'desc:"description of this environment"')
        self.Draw = LEDraw()
        self.SetTags("Draw", 'desc:"draws LEDs onto image"')
        self.Vis = Vis()
        self.SetTags("Vis", 'desc:"visual processing params"')
        self.MinLED = int()
        self.SetTags("MinLED", 'min:"0" max:"19" desc:"minimum LED number to draw (0-19)"')
        self.MaxLED = int()
        self.SetTags("MaxLED", 'min:"0" max:"19" desc:"maximum LED number to draw (0-19)"')
        self.CurLED = int()
        self.SetTags("CurLED", 'inactive:"+" desc:"current LED number that was drawn"')
        self.PrvLED = int()
        self.SetTags("PrvLED", 'inactive:"+" desc:"previous LED number that was drawn"')
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
        self.Output = etensor.Float32()
        self.SetTags("Output", 'desc:"CurLED one-hot output tensor"')

    def Name(ev):
        return ev.Nm

    def Desc(ev):
        return ev.Dsc

    def Validate(ev):
        return go.nil

    def Counters(ev):
        return []env.TimeScales(env.Run, env.Epoch, env.Sequence, env.Trial)

    def States(ev):
        isz = ev.Draw.ImgSize
        sz = ev.Vis.V1AllTsr.Shapes()
        nms = ev.Vis.V1AllTsr.DimNames()
        els = env.Elements(
            ("Image", go.Slice_int([isz.Y, isz.X]), []str("Y", "X")),
            ("V1", sz, nms),
            ("Output", go.Slice_int([4, 5]), []str("Y", "X")),
        )
        return els

    def State(ev, element):
        switch element:
        if "Image":
            vfilter.RGBToGrey(ev.Draw.Image, ev.OrigImg, 0, False) # pad for filt, bot zero
            return ev.OrigImg
        if "V1":
            return ev.Vis.V1AllTsr
        if "Output":
            return ev.Output
        return go.nil

    def Actions(ev):
        return go.nil

    def Defaults(ev):
        ev.Draw.Defaults()
        ev.Vis.Defaults()
        ev.XFormRand.TransX.Set(-0.25, 0.25)
        ev.XFormRand.TransY.Set(-0.25, 0.25)
        ev.XFormRand.Scale.Set(0.7, 1)
        ev.XFormRand.Rot.Set(-3.6, 3.6)

    def Init(ev, run):
        ev.Draw.Init()
        ev.Run.Scale = env.Run
        ev.Epoch.Scale = env.Epoch
        ev.Trial.Scale = env.Trial
        ev.Run.Init()
        ev.Epoch.Init()
        ev.Trial.Init()
        ev.Run.Cur = run
        ev.Trial.Cur = -1 # init state -- key so that first Step() = 0
        ev.Output.SetShape(go.Slice_int([4, 5]), go.nil, []str("Y", "X"))

    def Step(ev):
        ev.Epoch.Same()     # good idea to just reset all non-inner-most counters at start
        if ev.Trial.Incr(): # if true, hit max, reset to 0
            ev.Epoch.Incr()
        ev.DrawRndLED()
        ev.FilterImg()
        # debug only:
        # vfilter.RGBToGrey(ev.Draw.Image, &ev.OrigImg, 0, false) // pad for filt, bot zero
        return True

    def DoObject(ev, objno):
        """
        DoObject renders specific object (LED number)
        """
        ev.DrawLED(objno)
        ev.FilterImg()

    def Action(ev, element, input):

    def Counter(ev, scale):
        switch scale:
        if env.Run:
            return ev.Run.Query()
        if env.Epoch:
            return ev.Epoch.Query()
        if env.Trial:
            return ev.Trial.Query()
        return -1, -1, False

    def String(ev):
        """
        String returns the string rep of the LED env state
        """
        return "Obj: %02d, %s" % (ev.CurLED, ev.XForm.String())

    def SetOutput(ev, out):
        """
        SetOutput sets the output LED bit
        """
        ev.Output.SetZeros()
        ev.Output.SetFloat1D(out, 1)

    def DrawRndLED(ev):
        """
        DrawRndLED picks a new random LED and draws it
        """
        rng = 1 + ev.MaxLED - ev.MinLED
        led = ev.MinLED + rand.Intn(rng)
        ev.DrawLED(led)

    def DrawLED(ev, led):
        """
        DrawLED draw specified LED
        """
        ev.Draw.Clear()
        ev.Draw.DrawLED(led)
        ev.PrvLED = ev.CurLED
        ev.CurLED = led
        ev.SetOutput(ev.CurLED)

    def FilterImg(ev):
        """
        FilterImg filters the image from LED
        """
        ev.XFormRand.Gen(ev.XForm)
        img = ev.XForm.Image(ev.Draw.Image)
        ev.Vis.Filter(img)













# Compile-time check that implements Env interface
_ = (*LEDEnv)(go.nil)

