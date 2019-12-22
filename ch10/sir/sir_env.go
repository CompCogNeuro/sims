// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/kit"
)

// Actions are SIR actions
type Actions int

//go:generate stringer -type=Actions

var KiT_Actions = kit.Enums.AddEnum(ActionsN, kit.NotBitFlag, nil)

func (ev Actions) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Actions) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	Store Actions = iota
	Ignore
	Recall
	ActionsN
)

// SIREnv implements the store-ignore-recall task
type SIREnv struct {
	Nm       string          `desc:"name of this environment"`
	Dsc      string          `desc:"description of this environment"`
	NStim    int             `desc:"number of different stimuli that can be maintained"`
	RewVal   float32         `desc:"value for reward, based on whether model output = target"`
	NoRewVal float32         `desc:"value for non-reward"`
	Act      Actions         `desc:"current action"`
	Stim     int             `desc:"current stimulus"`
	Maint    int             `desc:"current stimulus being maintained"`
	Input    etensor.Float64 `desc:"input pattern with action + stim"`
	Output   etensor.Float64 `desc:"output pattern of what to respond"`
	Reward   etensor.Float64 `desc:"reward value"`
	Run      env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch    env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial    env.Ctr         `view:"inline" desc:"trial is the step counter within epoch"`
}

func (ev *SIREnv) Name() string { return ev.Nm }
func (ev *SIREnv) Desc() string { return ev.Dsc }

// SetNStim initializes env for given number of stimuli, init states
func (ev *SIREnv) SetNStim(n int) {
	ev.NStim = n
	ev.Input.SetShape([]int{int(ActionsN) + n}, nil, []string{"N"})
	ev.Output.SetShape([]int{n}, nil, []string{"N"})
	ev.Reward.SetShape([]int{1}, nil, []string{"1"})
	if ev.RewVal == 0 {
		ev.RewVal = 1
	}
}

func (ev *SIREnv) Validate() error {
	if ev.NStim <= 0 {
		return fmt.Errorf("SIREnv: %v NStim == 0 -- must set with SetNStim call", ev.Nm)
	}
	return nil
}

func (ev *SIREnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Trial}
}

func (ev *SIREnv) States() env.Elements {
	els := env.Elements{
		{"Input", []int{int(ActionsN) + ev.NStim}, []string{"N"}},
		{"Output", []int{ev.NStim}, []string{"N"}},
		{"Reward", []int{1}, nil},
	}
	return els
}

func (ev *SIREnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "Output":
		return &ev.Output
	case "Reward":
		return &ev.Reward
	}
	return nil
}

func (ev *SIREnv) Actions() env.Elements {
	return nil
}

// StimStr returns a letter string rep of stim (A, B...)
func (ev *SIREnv) StimStr(stim int) string {
	return string([]byte{byte('A' + stim)})
}

// String returns the current state as a string
func (ev *SIREnv) String() string {
	return fmt.Sprintf("%v_%v_mnt_%v_rew_%v", ev.Act, ev.StimStr(ev.Stim), ev.StimStr(ev.Maint), ev.Reward.Values[0])
}

func (ev *SIREnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.Maint = -1
}

// SetState sets the input, output states
func (ev *SIREnv) SetState() {
	ev.Input.SetZeros()
	ev.Input.Values[ev.Act] = 1
	if ev.Act != Recall {
		ev.Input.Values[int(ActionsN)+ev.Stim] = 1
	}
	ev.Output.SetZeros()
	ev.Output.Values[ev.Stim] = 1
}

// SetReward sets reward based on network's output
func (ev *SIREnv) SetReward(netout int) bool {
	cor := ev.Stim // already correct
	rw := netout == cor
	if rw {
		ev.Reward.Values[0] = float64(ev.RewVal)
	} else {
		ev.Reward.Values[0] = float64(ev.NoRewVal)
	}
	return rw
}

// Step the SIR task
func (ev *SIREnv) StepSIR() {
	for {
		ev.Act = Actions(rand.Intn(int(ActionsN)))
		if ev.Act == Store && ev.Maint >= 0 { // already full
			continue
		}
		if ev.Act == Recall && ev.Maint < 0 { // nothign
			continue
		}
		break
	}
	ev.Stim = rand.Intn(ev.NStim)
	switch ev.Act {
	case Store:
		ev.Maint = ev.Stim
	case Ignore:
	case Recall:
		ev.Stim = ev.Maint
		ev.Maint = -1
	}
	ev.SetState()
}

func (ev *SIREnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start

	ev.StepSIR()

	if ev.Trial.Incr() {
		ev.Epoch.Incr()
	}
	return true
}

func (ev *SIREnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *SIREnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*SIREnv)(nil)
