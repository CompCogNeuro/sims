// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/leabra/examples/pvlv/data"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

////////////////////////////////////////////////////////////////////////////////
// 	    Running the network..

type StepGrain int

const (
	Cycle StepGrain = iota
	Quarter
	AlphaMinus
	AlphaFull
	SGTrial    // Trial
	TrialBlock // Block
	Condition
	StepGrainN
)

var KiT_StepGrain = kit.Enums.AddEnum(StepGrainN, kit.NotBitFlag, nil)

func (ss *Sim) SettleMinus(train bool) {
	ev := &ss.Env
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	for qtr := 0; qtr < 3; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if ss.CycleLogUpdt == leabra.Cycle {
				ev.GlobalStep++
				ss.LogCycleData()
			}
			ss.Time.CycleInc()
			if ss.Stepper.StepPoint(int(Cycle)) {
				return
			}
			//ss.MaybeUpdate(train, false, leabra.FastSpike)
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView()
					}
				case leabra.FastSpike: // every 10 cycles
					if (cyc+1)%10 == 0 {
						ss.UpdateView()
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Quarter:
				ss.UpdateView()
			case leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView()
				}
			}
		}
		ss.Time.QuarterInc()
		if ss.CycleLogUpdt == leabra.Quarter {
			ev.GlobalStep++
			ss.LogCycleData()
		}
		if ss.Stepper.StepPoint(int(Quarter)) {
			return
		}
	}
}

func (ss *Sim) SettlePlus(train bool) {
	ev := &ss.Env
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
		ss.Net.Cycle(&ss.Time)
		if ss.CycleLogUpdt == leabra.Cycle {
			ev.GlobalStep++
			ss.LogCycleData()
		}
		ss.Time.CycleInc()
		if ss.Stepper.StepPoint(int(Cycle)) {
			return
		}
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Cycle:
				if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
					ss.UpdateView()
				}
			case leabra.FastSpike:
				if (cyc+1)%10 == 0 {
					ss.UpdateView()
				}
			}
		}
	}
	ss.Net.QuarterFinal(&ss.Time)
	if ss.ViewOn {
		switch viewUpdt {
		case leabra.Quarter, leabra.Phase:
			ss.UpdateView()
		}
	}
	ss.Time.QuarterInc()
	if ss.CycleLogUpdt == leabra.Quarter {
		ev.GlobalStep++
		ss.LogCycleData()
	}
	if ss.Stepper.StepPoint(int(Quarter)) {
		return
	}
}

func (ss *Sim) TrialStart(train bool) {
	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}
	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
}

func (ss *Sim) TrialEnd(_ *PVLVEnv, train bool) {
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	if ss.ViewOn && viewUpdt == leabra.Trial {
		ss.UpdateView()
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ev := &ss.Env
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"StimIn", "ContextIn", "USTimeIn"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := ev.State(ly.Nm)
		if pats == nil {
			continue
		}
		ly.ApplyExt(pats)
	}
}

func (ss *Sim) ApplyPVInputs() {
	ev := &ss.Env
	lays := []string{"PosPV", "NegPV"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := ev.State(ly.Nm)
		if pats == nil {
			continue
		}
		ly.ApplyExt(pats)
	}
}

// SingleTrial and functions -- SingleTrial has been consolidated into this
// A block is a set of trials, whose length is set by the current ConditionParams record
func (ev *PVLVEnv) RunOneTrialBlk(ss *Sim) {
	blockDone := false
	var curTG *data.TrialInstance
	ev.BlockStart(ss)
	ev.SetActiveTrialList(ss) // sets up one block's worth of data
	blockDone = ev.TrialCt.Cur >= ev.TrialCt.Max
	for !blockDone {
		if ev.TrialInstances.AtEnd() {
			panic(fmt.Sprintf("ran off end of TrialInstances list"))
		}
		curTG = ev.TrialInstances.ReadNext()
		ev.AlphaCycle.Max = curTG.AlphaTicksPerTrialGp
		blockDone = ev.RunOneTrial(ss, curTG) // run one instantiated trial type (aka "trial group")
		if ss.ViewOn && ss.TrainUpdt == leabra.Trial {
			ss.UpdateView()
		}
		if ss.Stepper.StepPoint(int(SGTrial)) {
			return
		}
	}
	ev.TrialBlockCt.Incr()
	ev.BlockEnded = true
	ev.BlockEnd(ss) // run monitoring and analysis, maybe save weights
	if ss.Stepper.StepPoint(int(TrialBlock)) {
		return
	}
	if ss.ViewOn && ss.TrainUpdt >= leabra.Epoch {
		ss.UpdateView()
	}
}

// run through a complete trial, consisting of a number of ticks as specified in the Trial spec
func (ev *PVLVEnv) RunOneTrial(ss *Sim, curTrial *data.TrialInstance) (blockDone bool) {
	var train bool
	trialDone := false
	ss.Net.ClearModActs(&ss.Time)
	for !trialDone {
		ev.SetupOneAlphaTrial(curTrial, 0)
		train = !ev.IsTestTrial(curTrial)
		ev.RunOneAlphaCycle(ss, curTrial)
		trialDone = ev.AlphaCycle.Incr()
		if ss.Stepper.StepPoint(int(AlphaFull)) {
			return
		}
		if ss.ViewOn && ss.TrainUpdt <= leabra.Quarter {
			ss.UpdateView()
		}
	}
	ss.Net.ClearMSNTraces(&ss.Time)
	blockDone = ev.TrialCt.Incr()
	ss.TrialEnd(ev, train)
	//ss.LogTrialData(ev) // accumulate
	if ss.ViewOn && ss.TrainUpdt == leabra.Trial {
		ss.UpdateView()
	}
	return blockDone
}

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ev *PVLVEnv) RunOneAlphaCycle(ss *Sim, trial *data.TrialInstance) {
	train := !ev.IsTestTrial(trial)
	ss.TrialStart(train)
	ev.SetState()
	ss.ApplyInputs()
	ss.SettleMinus(train)
	if ss.Stepper.StepPoint(int(AlphaMinus)) {
		return
	}
	ss.ApplyInputs()
	ss.ApplyPVInputs()
	ss.SettlePlus(train)
	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && ss.TrainUpdt == leabra.AlphaCycle {
		ss.UpdateView()
	}
	ss.LogTrialTypeData()
	//_ = ss.Stepper.StepPoint(int(AlphaPlus))
}

// brought over from cemer. This was named StepStopTest in cemer
func (ev *PVLVEnv) TrialNameStopTest(_ *Sim) bool {
	return false
}

// end SingleTrial and functions

// TrainEnd
func (ev *PVLVEnv) TrainEnd(ss *Sim) {
	if ev.CurConditionParams.SaveFinalWts {
		ev.SaveWeights(ss)
	}
	ss.Stepper.Stop()
}

// end TrainEnd
