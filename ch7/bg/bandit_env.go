// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/etable/etensor"
)

// BanditEnv simulates an n-armed bandit, where each of n inputs is associated with
// a specific probability of reward.
type BanditEnv struct {
	Nm       string          `desc:"name of this environment"`
	Dsc      string          `desc:"description of this environment"`
	N        int             `desc:"number of different inputs"`
	P        []float32       `desc:"no-inline" desc:"probabilities for each option"`
	RewVal   float32         `desc:"value for reward"`
	NoRewVal float32         `desc:"value for non-reward"`
	Option   env.CurPrvInt   `desc:"bandit option current / prev"`
	RndOpt   bool            `desc:"if true, select option at random each Step -- otherwise must be set externally (e.g., by model)"`
	Input    etensor.Float64 `desc:"one-hot input representation of current option"`
	Reward   etensor.Float64 `desc:"single reward value"`
	Run      env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch    env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial    env.Ctr         `view:"inline" desc:"trial is the step counter within epoch"`
}

func (ev *BanditEnv) Name() string { return ev.Nm }
func (ev *BanditEnv) Desc() string { return ev.Dsc }

// SetN initializes env for given number of options, and inits states
func (ev *BanditEnv) SetN(n int) {
	ev.N = n
	ev.P = make([]float32, n)
	ev.Input.SetShape([]int{n}, nil, []string{"N"})
	ev.Reward.SetShape([]int{1}, nil, []string{"1"})
	if ev.RewVal == 0 {
		ev.RewVal = 1
	}
}

func (ev *BanditEnv) Validate() error {
	if ev.N <= 0 {
		return fmt.Errorf("BanditEnv: %v N == 0 -- must set with Init call", ev.Nm)
	}
	return nil
}

func (ev *BanditEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Trial}
}

func (ev *BanditEnv) States() env.Elements {
	els := env.Elements{
		{"Input", []int{ev.N}, []string{"N"}}, // one-hot input
		{"Reward", []int{1}, nil},
	}
	return els
}

func (ev *BanditEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "Reward":
		return &ev.Reward
	}
	return nil
}

func (ev *BanditEnv) Actions() env.Elements {
	return nil
}

// String returns the current state as a string
func (ev *BanditEnv) String() string {
	return fmt.Sprintf("S_%d_%v", ev.Option.Cur, ev.Reward.Values[0])
}

func (ev *BanditEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.Option.Cur = 0
	ev.Option.Prv = -1
}

// RandomOpt selects option at random -- sets Option.Cur and returns it
func (ev *BanditEnv) RandomOpt() int {
	op := rand.Intn(ev.N)
	ev.Option.Set(op)
	return op
}

// SetInput sets the input state
func (ev *BanditEnv) SetInput() {
	ev.Input.SetZeros()
	ev.Input.Values[ev.Option.Cur] = 1
}

// SetReward sets reward for current option according to probability -- returns true if rewarded
func (ev *BanditEnv) SetReward() bool {
	p := ev.P[ev.Option.Cur]
	rw := erand.BoolP(p)
	if rw {
		ev.Reward.Values[0] = float64(ev.RewVal)
	} else {
		ev.Reward.Values[0] = float64(ev.NoRewVal)
	}
	return rw
}

func (ev *BanditEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start

	if ev.RndOpt {
		ev.RandomOpt()
	}
	ev.SetInput()
	ev.SetReward()

	if ev.Trial.Incr() {
		ev.Epoch.Incr()
	}
	return true
}

func (ev *BanditEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *BanditEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
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
var _ env.Env = (*BanditEnv)(nil)
