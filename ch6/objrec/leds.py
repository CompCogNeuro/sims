# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

package main

import "image"

"github.com/goki/gi/gi"

class LEDraw(pygiv.ClassViewObj):
# LEDraw renders old-school "LED" style "letters" composed of a set of horizontal
# and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
# Renders using SVG.
    """
    LEDraw renders old-school "LED" style "letters" composed of a set of horizontal
    and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
    Renders using SVG.
    """

    def __init__(self):
        super(Sim, self).__init__()
        self.Width = float()
        self.SetTags("Width", 'def:"4" desc:"line width of LEDraw as percent of display size"')
        self.Size = float()
        self.SetTags("Size", 'def:"0.6" desc:"size of overall LED as proportion of overall image size"')
        self.LineColor = gi.ColorName()
        self.SetTags("LineColor", 'desc:"color name for drawing lines"')
        self.BgColor = gi.ColorName()
        self.SetTags("BgColor", 'desc:"color name for background"')
        self.ImgSize = image.Point()
        self.SetTags("ImgSize", 'desc:"size of image to render"')
        self.Image = image.RGBA()
        self.SetTags("Image", 'view:"-" desc:"rendered image"')
        self.Paint = gi.Paint()
        self.SetTags("Paint", 'view:"+" desc:"painter object"')
        self.Render = gi.RenderState()
        self.SetTags("Render", 'view:"-" desc:"rendering state"')

    def Defaults(ld):
        ld.ImgSize = image.Point(120, 120)
        ld.Width = 4
        ld.Size = 0.6
        ld.LineColor = "white"
        ld.BgColor = "black"

    def Init(ld):
        """
        Init ensures that the image is created and of the right size, and renderer is initialized
        """
        if ld.ImgSize.X == 0 or ld.ImgSize.Y == 0:
            ld.Defaults()
        if ld.Image != go.nil:
            cs = ld.Image.Bounds().Size()
            if cs != ld.ImgSize:
                ld.Image = go.nil
        if ld.Image == go.nil:
            ld.Image = image.NewRGBA(image.Rectangle(Max= ld.ImgSize))
        ld.Render.Init(ld.ImgSize.X, ld.ImgSize.Y, ld.Image)
        ld.Paint.Defaults()
        ld.Paint.StrokeStyle.Width.SetPct(ld.Width)
        ld.Paint.StrokeStyle.Color.SetName(str(ld.LineColor))
        ld.Paint.FillStyle.Color.SetName(str(ld.BgColor))
        ld.Paint.SetUnitContextExt(ld.ImgSize)

    def Clear(ld):
        """
        Clear clears the image with BgColor
        """
        if ld.Image == go.nil:
            ld.Init()
        ld.Paint.Clear(ld.Render)

    def DrawSeg(ld, seg):
        """
        DrawSeg draws one segment
        """
        rs = ld.Render
        ctrX = float(ld.ImgSize.X) * 0.5
        ctrY = float(ld.ImgSize.Y) * 0.5
        szX = ctrX * ld.Size
        szY = ctrY * ld.Size

        switch seg:
        if Bottom.Bottom:
            ld.Paint.DrawLine(rs, ctrX-szX, ctrY+szY, ctrX+szX, ctrY+szY)
        if Left.Left:
            ld.Paint.DrawLine(rs, ctrX-szX, ctrY-szY, ctrX-szX, ctrY+szY)
        if Right.Right:
            ld.Paint.DrawLine(rs, ctrX+szX, ctrY-szY, ctrX+szX, ctrY+szY)
        if Top.Top:
            ld.Paint.DrawLine(rs, ctrX-szX, ctrY-szY, ctrX+szX, ctrY-szY)
        if CenterH.CenterH:
            ld.Paint.DrawLine(rs, ctrX-szX, ctrY, ctrX+szX, ctrY)
        if CenterV.CenterV:
            ld.Paint.DrawLine(rs, ctrX, ctrY-szY, ctrX, ctrY+szY)
        ld.Paint.Stroke(rs)

    def DrawLED(ld, num):
        """
        DrawLED draws one LED of given number, based on LEDdata
        """
        led = LEData[num]
        for _, seg in led :
            ld.DrawSeg(seg)




class LEDSegs(pygiv.ClassViewObj):


Bottom.Bottom LEDSegs = iota
Left.Left
Right.Right
Top.Top
CenterH.CenterH
CenterV.CenterV
LEDSegsN.LEDSegsN

LEData = [][3]LEDSegs(
    (CenterH.CenterH, CenterV.CenterV, Right.Right),
    (Top.Top, CenterV.CenterV, Bottom.Bottom),
    (Top.Top, Right.Right, Bottom.Bottom),
    (Bottom.Bottom, CenterV.CenterV, Right.Right),
    (Left.Left, CenterH.CenterH, Right.Right),

    (Left.Left, CenterV.CenterV, CenterH.CenterH),
    (Left.Left, CenterV.CenterV, Right.Right),
    (Left.Left, CenterV.CenterV, Bottom.Bottom),
    (Left.Left, CenterH.CenterH, Top.Top),
    (Left.Left, CenterH.CenterH, Bottom.Bottom),

    (Top.Top, CenterV.CenterV, Right.Right),
    (Bottom.Bottom, CenterV.CenterV, CenterH.CenterH),
    (Right.Right, CenterH.CenterH, Bottom.Bottom),
    (Top.Top, CenterH.CenterH, Bottom.Bottom),
    (Left.Left, Top.Top, Right.Right),

    (Top.Top, CenterH.CenterH, Right.Right),
    (Left.Left, CenterV.CenterV, Top.Top),
    (Top.Top, Left.Left, Bottom.Bottom),
    (Left.Left, Bottom.Bottom, Right.Right),
    (Top.Top, CenterV.CenterV, CenterH.CenterH),
)
