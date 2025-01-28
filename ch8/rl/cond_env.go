// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"cogentcore.org/lab/base/randx"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/etensor/tensor"
)

// OnOff represents stimulus On / Off timing
type OnOff struct {

	// is this stimulus active -- use it?
	Act bool

	// when stimulus turns on
	On int

	// when stimulu turns off
	Off int

	// probability of being active on any given trial
	P float32

	// variability in onset timing (max number of trials before/after On that it could start)
	OnVar int

	// variability in offset timing (max number of trials before/after Off that it could end)
	OffVar int

	// current active status based on P probability
	CurAct bool `display:"-"`

	// current on / off values using Var variability
	CurOn, CurOff int `display:"-"`
}

func (oo *OnOff) Set(act bool, on, off int) {
	oo.Act = act
	oo.On = on
	oo.Off = off
	oo.P = 1 // default
}

// TrialUpdate updates Cur state at start of trial
func (oo *OnOff) TrialUpdate() {
	if !oo.Act {
		return
	}
	oo.CurAct = randx.BoolP(float64(oo.P))
	oo.CurOn = oo.On - oo.OnVar + 2*rand.Intn(oo.OnVar+1)
	oo.CurOff = oo.Off - oo.OffVar + 2*rand.Intn(oo.OffVar+1)
}

// IsOn returns true if should be on according current time
func (oo *OnOff) IsOn(tm int) bool {
	return oo.Act && oo.CurAct && tm >= oo.CurOn && tm < oo.CurOff
}

// CondEnv simulates an n-armed bandit, where each of n inputs is associated with
// a specific probability of reward.
type CondEnv struct {

	// name of this environment
	Name string

	// total time for trial
	TotTime int

	// Conditioned stimulus A (e.g., Tone)
	CSA OnOff `display:"inline"`

	// Conditioned stimulus B (e.g., Light)
	CSB OnOff `display:"inline"`

	// Conditioned stimulus C
	CSC OnOff `display:"inline"`

	// Unconditioned stimulus -- reward
	US OnOff `display:"inline"`

	// value for reward
	RewVal float32

	// value for non-reward
	NoRewVal float32

	// one-hot input representation of current option
	Input tensor.Float64

	// single reward value
	Reward tensor.Float64

	// one trial is a pass through all TotTime Events
	Trial env.Counter `display:"inline"`

	// event is one time step within Trial -- e.g., CS turning on, etc
	Event env.Counter `display:"inline"`
}

func (ev *CondEnv) Label() string { return ev.Name }

func (ev *CondEnv) Defaults() {
	ev.TotTime = 20
	ev.CSA.Set(true, 10, 16)
	ev.CSB.Set(false, 2, 10)
	ev.CSC.Set(false, 2, 5)
	ev.US.Set(true, 15, 16)
}

func (ev *CondEnv) State(element string) tensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "Rew":
		return &ev.Reward
	}
	return nil
}

// String returns the current state as a string
func (ev *CondEnv) String() string {
	return fmt.Sprintf("S_%d_%g", ev.Event.Cur, ev.Reward.Values[0])
}

func (ev *CondEnv) Init(run int) {
	ev.Input.SetShape([]int{3, ev.TotTime})
	ev.Reward.SetShape([]int{1})
	ev.Trial.Scale = etime.Trial
	ev.Event.Scale = etime.Event
	ev.Trial.Init()
	ev.Event.Init()
	ev.Event.Max = ev.TotTime
	ev.Event.Cur = -1 // init state -- key so that first Step() = 0
	ev.TrialUpdate()
}

// TrialUpdate updates all random vars at start of trial
func (ev *CondEnv) TrialUpdate() {
	ev.CSA.TrialUpdate()
	ev.CSB.TrialUpdate()
	ev.CSC.TrialUpdate()
	ev.US.TrialUpdate()
}

// SetInput sets the input state
func (ev *CondEnv) SetInput() {
	ev.Input.SetZeros()
	tm := ev.Event.Cur
	if ev.CSA.IsOn(tm) {
		ev.Input.Values[tm] = 1
	}
	if ev.CSB.IsOn(tm) {
		ev.Input.Values[ev.TotTime+tm] = 1
	}
	if ev.CSC.IsOn(tm) {
		ev.Input.Values[2*ev.TotTime+tm] = 1
	}
}

// SetReward sets reward for current option according to probability -- returns true if rewarded
func (ev *CondEnv) SetReward() bool {
	tm := ev.Event.Cur
	rw := ev.US.IsOn(tm)
	if rw {
		ev.Reward.Values[0] = float64(ev.RewVal)
	} else {
		ev.Reward.Values[0] = float64(ev.NoRewVal)
	}
	return rw
}

func (ev *CondEnv) Step() bool {
	ev.Trial.Same() // this ensures that they only report changed when actually changed

	incr := ev.Event.Incr()
	ev.SetInput()
	ev.SetReward()

	if incr {
		ev.TrialUpdate()
		ev.Trial.Incr()
	}
	return true
}

func (ev *CondEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*CondEnv)(nil)
