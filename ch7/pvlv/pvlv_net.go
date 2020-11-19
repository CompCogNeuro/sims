// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/leabra/pvlv"
	"log"
)

func (ss *Sim) ConfigNet(net *pvlv.Network) {
	var vtaPSendsTo, vtaNSendsTo []string

	net.InitName(net, "PVLV")

	// Order of creation is partly dictated by desired layout in the display

	// Inputs
	stimIn := net.AddLayer2D("StimIn", 12, 1, emer.Input)
	ctxIn := net.AddLayer2D("ContextIn", 20, 3, emer.Input)
	ustimeIn := net.AddLayer4D("USTimeIn", 16, 2, 4, 5, emer.Input)
	//

	// Primary value
	posPV := pvlv.AddPVLayer(net, "PosPV", 1, 4, emer.Input)
	posPV.SetClass("PVLVLayer PVLayer")
	negPV := pvlv.AddPVLayer(net, "NegPV", 1, 4, emer.Input)
	negPV.SetClass("PVLVLayer PVLayer")

	pptg := pvlv.AddPPTgLayer(net, "PPTg", 1, 1)

	lhbRmtG := pvlv.AddLHbRMTgLayer(net, "LHbRMTg") // Lateral habenula & Rostromedial tegmental nucleus

	cEmPos := net.AddLayer4D("CEmPos", 1, 4, 1, 1, emer.Hidden)
	cEmPos.SetClass("CEmLayer")
	cEmNeg := net.AddLayer4D("CEmNeg", 1, 4, 1, 1, emer.Hidden)
	cEmNeg.SetClass("CEmLayer")

	// Ventral Striatum (VS)

	// Ventral Striatum Patch (delayed, PV), Direct = Positive Valence expectation
	// (increases + on LHB, causing more dipping unless counteracted with PV+)
	vsPatchPosD1 := net.AddMSNLayer(
		"VSPatchPosD1", 1, 4, 1, 1, pvlv.PATCH, pvlv.D1R)
	vsPatchPosD1.SetClass("VSPatchLayer")
	vsPatchNegD2 := net.AddMSNLayer(
		"VSPatchNegD2", 1, 4, 1, 1, pvlv.PATCH, pvlv.D2R)
	vsPatchNegD2.SetClass("VSPatchLayer")

	// Ventral Striatum Patch (delayed, PV), indirect = Neg valence expectation
	// (removes + on LHb dipper -- to cancel PV- neg outcome)
	vsPatchPosD2 := net.AddMSNLayer(
		"VSPatchPosD2", 1, 4, 1, 1, pvlv.PATCH, pvlv.D2R)
	vsPatchPosD2.SetClass("VSPatchLayer")
	vsPatchNegD1 := net.AddMSNLayer(
		"VSPatchNegD1", 1, 4, 1, 1, pvlv.PATCH, pvlv.D1R)
	vsPatchNegD1.SetClass("VSPatchLayer")

	// Ventral Striatum Matrix (immediate, LV), Direct = Positive Valence (direct inhib of gpi, removes + on LHb dipper)
	vsMatrixPosD1 := net.AddMSNLayer(
		"VSMatrixPosD1", 1, 4, 1, 1, pvlv.MATRIX, pvlv.D1R)
	vsMatrixPosD1.SetClass("VSMatrixLayer")
	vsMatrixNegD2 := net.AddMSNLayer(
		"VSMatrixNegD2", 1, 4, 1, 1, pvlv.MATRIX, pvlv.D2R)
	vsMatrixNegD2.SetClass("VSMatrixLayer")

	// Ventral Striatum Matrix (immediate, LV), Indirect = Negative Valence (increases + on LHb, causing more dipping)
	vsMatrixPosD2 := net.AddMSNLayer(
		"VSMatrixPosD2", 1, 4, 1, 1, pvlv.MATRIX, pvlv.D2R)
	vsMatrixPosD2.SetClass("VSMatrixLayer")
	vsMatrixNegD1 := net.AddMSNLayer(
		"VSMatrixNegD1", 1, 4, 1, 1, pvlv.MATRIX, pvlv.D1R)
	vsMatrixNegD1.SetClass("VSMatrixLayer")

	for _, layer := range []*pvlv.MSNLayer{
		vsPatchPosD1, vsPatchPosD2, vsPatchNegD1, vsPatchNegD2,
		vsMatrixPosD1, vsMatrixPosD2, vsMatrixNegD1, vsMatrixNegD2} {
		layer.SetClass("VS")
		vtaPSendsTo = append(vtaPSendsTo, layer.Name())
	}

	// Learned Value
	// Amygdala
	// Basolateral amygdala (BLA)
	blAmygPosD1 := net.AddBlAmygLayer("BLAmygPosD1", 1, 4, 7, 9, pvlv.POS, pvlv.D1R, emer.Hidden)
	blAmygPosD2 := net.AddBlAmygLayer("BLAmygPosD2", 1, 4, 7, 9, pvlv.POS, pvlv.D2R, emer.Hidden)
	blAmygNegD1 := net.AddBlAmygLayer("BLAmygNegD1", 1, 4, 7, 9, pvlv.NEG, pvlv.D1R, emer.Hidden)
	blAmygNegD2 := net.AddBlAmygLayer("BLAmygNegD2", 1, 4, 7, 9, pvlv.NEG, pvlv.D2R, emer.Hidden)
	for _, layer := range []emer.Layer{blAmygPosD1, blAmygPosD2, blAmygNegD1, blAmygNegD2} {
		layer.SetClass("BLAmygLayer")
		vtaPSendsTo = append(vtaPSendsTo, layer.Name())
	}

	// Centrolateral amygdala
	celAcqPosD1 := net.AddCElAmygLayer("CElAcqPosD1", 1, 4, 1, 1, pvlv.Acq, pvlv.POS, pvlv.D1R)
	celExtPosD2 := net.AddCElAmygLayer("CElExtPosD2", 1, 4, 1, 1, pvlv.Ext, pvlv.POS, pvlv.D2R)
	celExtNegD1 := net.AddCElAmygLayer("CElExtNegD1", 1, 4, 1, 1, pvlv.Ext, pvlv.NEG, pvlv.D1R)
	celAcqNegD2 := net.AddCElAmygLayer("CElAcqNegD2", 1, 4, 1, 1, pvlv.Acq, pvlv.NEG, pvlv.D2R)
	for _, layer := range []emer.Layer{celAcqPosD1, celExtPosD2, celExtNegD1, celAcqNegD2} {
		layer.SetClass("CElAmyg")
		vtaPSendsTo = append(vtaPSendsTo, layer.Name())
	}

	vtaP := net.AddVTALayer("VTAp", pvlv.POS)
	vtaP.SetClass("PVLVLayer DALayer")

	vtaN := net.AddVTALayer("VTAn", pvlv.NEG)
	vtaN.SetClass("PVLVLayer DALayer")

	for _, rcvr := range vtaPSendsTo {
		vtaP.SendDA.Add(rcvr)
	}
	// leaving this in because it's there in the CEmer model. Flagged by GoLand because it does nothing
	for _, rcvr := range vtaNSendsTo {
		vtaN.SendDA.Add(rcvr)
	}

	// brain-dead assignment of threads to layers. On a 6-core Macbook Pro, gives about a 35% speedup
	if ss.LayerThreads {
		for i, ly := range net.Layers {
			ly.SetThread(i)
		}
	}

	// Connect everything
	pjFull := prjn.NewFull()
	pjPools := prjn.NewPoolOneToOne()

	// to BLAmygPosD1
	pj := net.ConnectLayersPrjn(posPV, blAmygPosD1, pjPools, emer.Forward, &pvlv.AmygModPrjn{}) // LR == 0, could probably just be a normal fixed projection
	pj.SetClass("PVLVLrnCons BLAmygConsUS")
	pj = net.ConnectLayersPrjn(stimIn, blAmygPosD1, pjFull, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("PVLVLrnCons BLAmygConsStim")
	pj = net.ConnectLayers(blAmygPosD2, blAmygPosD1, pjFull, emer.Inhib)
	pj.SetClass("PVLVLrnCons BLAmygConsInhib")
	blAmygPosD1.ILI.Lays.Add(blAmygNegD2.Name())

	// to BLAmygNegD2
	pj = net.ConnectLayersPrjn(negPV, blAmygNegD2, pjPools, emer.Forward, &pvlv.AmygModPrjn{}) // LR == 0
	pj.SetClass("PVLVLrnCons BLAmygConsUS")
	pj = net.ConnectLayersPrjn(stimIn, blAmygNegD2, pjFull, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("PVLVLrnCons BLAmygConsStim")
	pj = net.ConnectLayers(blAmygNegD1, blAmygNegD2, pjFull, emer.Inhib)
	pj.SetClass("PVLVLrnCons BLAmygConsInhib")
	blAmygNegD2.ILI.Lays.Add(blAmygPosD1.Name())

	// to BLAmygPosD2
	pj = net.ConnectLayersPrjn(ctxIn, blAmygPosD2, pjFull, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("PVLVLrnCons BLAmygConsCntxtExt")
	net.ConnectLayersActMod(blAmygPosD1, blAmygPosD2, 0.2)

	// to BLAmygNegD1
	pj = net.ConnectLayersPrjn(ctxIn, blAmygNegD1, pjFull, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("PVLVLrnCons BLAmygConsCntxtExt")
	net.ConnectLayersActMod(blAmygNegD2, blAmygNegD1, 0.2)

	// to CElAcqPosD1
	pj = net.ConnectLayers(celExtPosD2, celAcqPosD1, pjPools, emer.Inhib) //
	pj.SetClass("CElExtToAcqInhib")
	posPV.AddPVReceiver(celAcqPosD1.Nm)
	pj = net.ConnectLayersPrjn(stimIn, celAcqPosD1, pjFull, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("CElAmygCons")
	//pj = net.ConnectLayers(stimIn, celAcqPosD1, pjFull, emer.Forward) // in CEmer, but off
	pj = net.ConnectLayersPrjn(blAmygPosD1, celAcqPosD1, pjPools, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("CElAmygConsFmBLA")

	// to CElAcqNegD2
	pj = net.ConnectLayers(celExtNegD1, celAcqNegD2, pjPools, emer.Inhib)
	pj.SetClass("CElExtToAcqInhib")
	negPV.AddPVReceiver(celAcqNegD2.Nm)
	pj = net.ConnectLayersPrjn(stimIn, celAcqNegD2, pjFull, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("CElAmygCons")
	//pj = net.ConnectLayers(stimIn, celAcqNegD2, pjFull, emer.Forward) // in CEmer, but off
	pj = net.ConnectLayersPrjn(blAmygNegD2, celAcqNegD2, pjPools, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("CElAmygConsFmBLA")

	// to CElExtPosD2
	pj = net.ConnectLayers(celAcqPosD1, celExtPosD2, pjPools, emer.Inhib)
	pj.SetClass("CElAcqToExtInhib")
	net.ConnectLayersActMod(celAcqPosD1, celExtPosD2, 1)
	pj = net.ConnectLayersPrjn(blAmygPosD2, celExtPosD2, pjPools, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("CElAmygConsExtFmBLA")

	// to CElExtNegD1
	pj = net.ConnectLayers(celAcqNegD2, celExtNegD1, pjPools, emer.Inhib)
	pj.SetClass("CElAcqToExtInhib")
	net.ConnectLayersActMod(celAcqNegD2, celExtNegD1, 1)
	pj = net.ConnectLayersPrjn(blAmygNegD1, celExtNegD1, pjPools, emer.Forward, &pvlv.AmygModPrjn{})
	pj.SetClass("CElAmygConsExtFmBLA")

	// to CEmPos
	pj = net.ConnectLayers(celAcqPosD1, cEmPos, pjPools, emer.Forward)
	pj.SetClass("CEltoCeMFixed")
	pj = net.ConnectLayers(celExtPosD2, cEmPos, pjPools, emer.Inhib)
	pj.SetClass("CEltoCeMFixed")

	// to CEmNeg
	pj = net.ConnectLayers(celAcqNegD2, cEmNeg, pjPools, emer.Forward)
	pj.SetClass("CEltoCeMFixed")
	pj = net.ConnectLayers(celExtNegD1, cEmNeg, pjPools, emer.Inhib)
	pj.SetClass("CEltoCeMFixed")

	// to VSPatchPosD1
	net.ConnectLayersActMod(blAmygPosD1, vsPatchPosD1, 0.2)
	pj = net.ConnectLayersPrjn(ustimeIn, vsPatchPosD1, pjFull, emer.Forward,
		&pvlv.MSNPrjn{LearningRule: pvlv.DAHebbVS})
	pj.SetClass("PVLVLrnCons VSPatchConsToPosD1")

	// to VSPatchPosD2
	net.ConnectLayersActMod(blAmygPosD1, vsPatchPosD2, 0.2)
	pj = net.ConnectLayersPrjn(ustimeIn, vsPatchPosD2, pjFull, emer.Forward,
		&pvlv.MSNPrjn{LearningRule: pvlv.DAHebbVS})
	pj.SetClass("PVLVLrnCons VSPatchConsToPosD2")

	// to VSPatchNegD2
	net.ConnectLayersActMod(blAmygNegD2, vsPatchNegD2, 0.2)
	pj = net.ConnectLayersPrjn(ustimeIn, vsPatchNegD2, pjFull, emer.Forward,
		&pvlv.MSNPrjn{LearningRule: pvlv.DAHebbVS})
	pj.SetClass("PVLVLrnCons VSPatchConsToNegD2")

	// to VSPatchNegD1
	net.ConnectLayersActMod(blAmygNegD2, vsPatchNegD1, 0.2)
	pj = net.ConnectLayersPrjn(ustimeIn, vsPatchNegD1, pjFull, emer.Forward,
		&pvlv.MSNPrjn{LearningRule: pvlv.DAHebbVS})
	pj.SetClass("PVLVLrnCons VSPatchConsToNegD1")

	// to VSMatrixPosD1
	net.ConnectLayersActMod(blAmygPosD1, vsMatrixPosD1, 0.015)
	pj = net.ConnectLayersPrjn(stimIn, vsMatrixPosD1, pjFull, emer.Forward,
		&pvlv.MSNPrjn{LearningRule: pvlv.TraceNoThalVS})
	pj.SetClass("PVLVLrnCons VSMatrixConsToPosD1")

	// to VSMatrixPosD2
	net.ConnectLayersActMod(vsMatrixPosD1, vsMatrixPosD2, 1)
	pj = net.ConnectLayersPrjn(stimIn, vsMatrixPosD2, pjFull, emer.Forward,
		&pvlv.MSNPrjn{LearningRule: pvlv.TraceNoThalVS})
	pj.SetClass("PVLVLrnCons VSMatrixConsToPosD2")

	// to VSMatrixNegD2
	net.ConnectLayersActMod(blAmygNegD2, vsMatrixNegD2, 0.015)
	pj = net.ConnectLayersPrjn(stimIn, vsMatrixNegD2, pjFull, emer.Forward,
		&pvlv.MSNPrjn{LearningRule: pvlv.TraceNoThalVS})
	pj.SetClass("PVLVLrnCons VSMatrixConsToNegD2")

	// to VSMatrixNegD1
	net.ConnectLayersActMod(vsMatrixNegD2, vsMatrixNegD1, 1)
	pj = net.ConnectLayersPrjn(stimIn, vsMatrixNegD1, pjFull, emer.Forward,
		&pvlv.MSNPrjn{LearningRule: pvlv.TraceNoThalVS})
	pj.SetClass("PVLVLrnCons VSMatrixConsToNegD1")

	// to PPTg
	pj = net.ConnectLayers(cEmPos, pptg, pjFull, emer.Forward)
	pj.SetClass("PVLVFixedCons")

	// LHBRMTg sources
	for _, ly := range []emer.Layer{
		posPV, negPV, vtaP, vtaN,
		vsPatchPosD1, vsPatchPosD2, vsPatchNegD1, vsPatchNegD2,
		vsMatrixPosD1, vsMatrixPosD2, vsMatrixNegD1, vsMatrixNegD2} {
		lhbRmtG.RcvFrom.Add(ly.Name())
	}

	// "Marker" connections (in the cemer world)
	// VTAp -> BLAmygPosD1, BLAmygPosD2, BLAmygNegD1, BLAmygNegD2,
	//         CElAcqPosD1, CElExtPosD2, CElAcqNegD1, CElAcqNegD2,
	//         VSPatchPosD1, VSPatchPosD2, VSMatrixNegD1, VSMatrixNegD2
	// LHbRMTg <- PosPV, NegPV,
	//            VSPatchPosD1, VSPatchPosD2, VSPatchNegD1, VSPatchNegD2,
	//            VSMatrixPosD1, VSMatrixPosD2, VSMatrixNegD1, VSMatrixNegD2
	// VTAp <- PPTg, LHbRMTg, PosPV, VSPatchPosD1, VSPatchPosD2, VSPatchNegD1, VSPatchNegD2
	// VTAn <- PPTg, LHbRMTg, NegPV, VSPatchNegD1, VSPatchNegD2

	// Lay out for display

	stimIn.SetRelPos(relpos.Rel{Scale: 3})
	ctxIn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "StimIn", Space: 8, Scale: 3})
	ustimeIn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ContextIn", Space: 6, Scale: 3})

	negPV.SetRelPos(relpos.Rel{Rel: relpos.Below, Other: stimIn.Name(), Scale: 3})
	posPV.SetRelPos(relpos.Rel{Rel: relpos.LeftOf, Other: negPV.Name(), Space: 4, XAlign: relpos.Left, Scale: 3})
	pptg.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: negPV.Name(), Space: 3, Scale: 3})
	vtaP.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: pptg.Name(), Space: 4, Scale: 3})
	lhbRmtG.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: vtaP.Name(), Space: 4, Scale: 3})
	vtaN.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: lhbRmtG.Name(), Space: 10, Scale: 3})

	cEmPos.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: posPV.Name(), Space: 8, Scale: 3})
	cEmNeg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: negPV.Name(), Space: 8, Scale: 3})

	celAcqPosD1.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: cEmPos.Name(), Space: 3, Scale: 3})
	celExtPosD2.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: celAcqPosD1.Name(), Space: 3, Scale: 3})

	celExtNegD1.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: cEmNeg.Name(), Space: 3, Scale: 3})
	celAcqNegD2.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: celExtNegD1.Name(), Space: 3, Scale: 3})

	vsPatchPosD1.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: lhbRmtG.Name(), Space: 6, Scale: 3})
	vsPatchPosD2.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vsPatchPosD1.Name(), Space: 2, Scale: 3})
	vsMatrixPosD1.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vsPatchPosD2.Name(), Space: 4, Scale: 3})
	vsMatrixPosD2.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vsMatrixPosD1.Name(), Space: 2, Scale: 3})

	vsPatchNegD1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: vsPatchPosD1.Name(), Space: 2, Scale: 3})
	vsPatchNegD2.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vsPatchNegD1.Name(), Space: 2, Scale: 3})
	vsMatrixNegD1.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vsPatchNegD2.Name(), Space: 4, Scale: 3})
	vsMatrixNegD2.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: vsMatrixNegD1.Name(), Space: 2, Scale: 3})

	blAmygPosD1.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: celExtPosD2.Name(), Space: 10})
	blAmygPosD2.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: blAmygPosD1.Name(), Space: 3})

	blAmygNegD2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: blAmygPosD1.Name(), Space: 6})
	blAmygNegD1.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: blAmygNegD2.Name(), Space: 3})
	//net.Layout()

	err := ss.SetParams("Network", false) // only set Network params
	if err != nil {
		log.Println(err)
	}
	net.Defaults()
	err = net.Build()
	if err != nil {
		log.Println(err)
		return
	}

	net.InitWts()

	//lowerLayers := []emer.Layer{negPV, posPV, pptg, vtaP, lhbRmtG, vtaN, cEmPos, cEmNeg, celAcqPosD1, celExtPosD2,
	//	celExtNegD1, celExtNegD1, celAcqNegD2, vsPatchPosD1, vsPatchPosD2, vsMatrixPosD1, vsMatrixPosD2,
	//	vsPatchNegD1, vsPatchNegD2, vsMatrixNegD1, vsMatrixNegD2, blAmygPosD1, blAmygPosD2,
	//	blAmygNegD2, blAmygNegD1}
	//for _, layer := range lowerLayers {
	//	pos := layer.Pos()
	//	pos.Z = 0.0
	//	layer.SetPos(pos)
	//}

	for _, layer := range []emer.Layer{stimIn, ctxIn, ustimeIn} {
		pos := layer.Pos()
		pos.Z = 0.5
		layer.SetPos(pos)
	}
	net.Layout()

}
