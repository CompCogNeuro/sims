# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, pygiv, gi, image, gist, girl
from enum import Enum

Bottom = 0
Left = 1
Right = 2
Top = 3
CenterH = 4
CenterV = 5
LEDSegsN = 6


class LEDSegs(Enum):
    Bottom = 0
    Left = 1
    Right = 2
    Top = 3
    CenterH = 4
    CenterV = 5
    LEDSegsN = 6


# These are the 20 different LED figures
LEData = (
    (CenterH, CenterV, Right),
    (Top, CenterV, Bottom),
    (Top, Right, Bottom),
    (Bottom, CenterV, Right),
    (Left, CenterH, Right),
    (Left, CenterV, CenterH),
    (Left, CenterV, Right),
    (Left, CenterV, Bottom),
    (Left, CenterH, Top),
    (Left, CenterH, Bottom),
    (Top, CenterV, Right),
    (Bottom, CenterV, CenterH),
    (Right, CenterH, Bottom),
    (Top, CenterH, Bottom),
    (Left, Top, Right),
    (Top, CenterH, Right),
    (Left, CenterV, Top),
    (Top, Left, Bottom),
    (Left, Bottom, Right),
    (Top, CenterV, CenterH),
)


class LEDraw(pyviews.ClassViewObj):
    """
    LEDraw renders old-school "LED" style "letters" composed of a set of horizontal
    and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
    Renders using SVG.
    """

    def __init__(self):
        super(LEDraw, self).__init__()
        self.Width = float(4)
        self.SetTags(
            "Width", 'def:"4" desc:"line width of LEDraw as percent of display size"'
        )
        self.Size = float(0.6)
        self.SetTags(
            "Size",
            'def:"0.6" desc:"size of overall LED as proportion of overall image size"',
        )
        self.LineColor = "white"
        self.SetTags("LineColor", 'desc:"color name for drawing lines"')
        self.BgColor = "black"
        self.SetTags("BgColor", 'desc:"color name for background"')
        self.ImgSize = image.Point()
        self.SetTags("ImgSize", 'desc:"size of image to render"')
        self.Image = image.RGBA()
        self.SetTags("Image", 'view:"-" desc:"rendered image"')
        self.Paint = girl.Paint()
        self.SetTags("Paint", 'view:"+" desc:"painter object"')
        self.Render = girl.State()
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
        if ld.Image != 0:
            cs = ld.Image.Bounds().Size()
            if cs != ld.ImgSize:
                ld.Image = 0
        if ld.Image == 0:
            ld.Image = image.NewRGBA(image.Rectangle(Max=ld.ImgSize))
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
        if ld.Image == 0:
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

        if seg == Bottom:
            ld.Paint.DrawLine(rs, ctrX - szX, ctrY + szY, ctrX + szX, ctrY + szY)
        if seg == Left:
            ld.Paint.DrawLine(rs, ctrX - szX, ctrY - szY, ctrX - szX, ctrY + szY)
        if seg == Right:
            ld.Paint.DrawLine(rs, ctrX + szX, ctrY - szY, ctrX + szX, ctrY + szY)
        if seg == Top:
            ld.Paint.DrawLine(rs, ctrX - szX, ctrY - szY, ctrX + szX, ctrY - szY)
        if seg == CenterH:
            ld.Paint.DrawLine(rs, ctrX - szX, ctrY, ctrX + szX, ctrY)
        if seg == CenterV:
            ld.Paint.DrawLine(rs, ctrX, ctrY - szY, ctrX, ctrY + szY)
        ld.Paint.Stroke(rs)

    def DrawLED(ld, num):
        """
        DrawLED draws one LED of given number, based on LEDdata
        """
        led = LEData[num]
        for seg in led:
            ld.DrawSeg(seg)
