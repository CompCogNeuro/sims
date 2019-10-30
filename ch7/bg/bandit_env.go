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
	Nm     string          `desc:"name of this environment"`
	Dsc    string          `desc:"description of this environment"`
	N      int             `desc:"number of different inputs"`
	P      []float32       `view:"no-inline" desc:"probabilities for each option"`
	Option env.CurPrvInt   `desc:"bandit option current / prev"`
	RndOpt bool            `desc:"if true, select option at random each Step -- otherwise must be set externally (e.g., by model)"`
	Input  etensor.Float64 `desc:"one-hot input representation of current option"`
	Reward etensor.Float64 `desc:"single reward value"`
	Run    env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch  env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial  env.Ctr         `view:"inline" desc:"trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence"`
}

func (be *BanditEnv) Name() string { return be.Nm }
func (be *BanditEnv) Desc() string { return be.Dsc }

// SetN initializes env for given number of options, and inits states
func (be *BanditEnv) SetN(n int) {
	be.N = n
	be.P = make([]float32, n)
	be.Input.SetShape([]int{n}, nil, []string{"N"})
	be.Reward.SetShape([]int{1}, nil, []string{"1"})
}

func (be *BanditEnv) Validate() error {
	if be.N <= 0 {
		return fmt.Errorf("BanditEnv: %v N == 0 -- must set with Init call", be.Nm)
	}
	return nil
}

func (be *BanditEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Trial}
}

func (be *BanditEnv) States() env.Elements {
	els := env.Elements{
		{"Input", []int{be.N}, []string{"N"}}, // one-hot input
		{"Reward", []int{1}, nil},
	}
	return els
}

func (be *BanditEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &be.Input
	case "Reward":
		return &be.Reward
	}
	return nil
}

func (be *BanditEnv) Actions() env.Elements {
	return nil
}

// String returns the current state as a string
func (be *BanditEnv) String() string {
	return fmt.Sprintf("S_%d_%v", be.Option.Cur, be.Reward.Values[0])
}

func (be *BanditEnv) Init(run int) {
	be.Run.Scale = env.Run
	be.Epoch.Scale = env.Epoch
	be.Trial.Scale = env.Trial
	be.Run.Init()
	be.Epoch.Init()
	be.Trial.Init()
	be.Run.Cur = run
	be.Trial.Max = 0
	be.Trial.Cur = -1 // init state -- key so that first Step() = 0
	be.Option.Cur = 0
	be.Option.Prv = -1
}

// RandomOpt selects option at random -- sets Option.Cur and returns it
func (be *BanditEnv) RandomOpt() int {
	op := rand.Intn(be.N)
	be.Option.Set(op)
	return op
}

// SetInput sets the input state
func (be *BanditEnv) SetInput() {
	be.Input.SetZeros()
	be.Input.Values[be.Option.Cur] = 1
}

// SetReward sets reward for current option according to probability -- returns true if rewarded
func (be *BanditEnv) SetReward() bool {
	p := be.P[be.Option.Cur]
	rw := erand.BoolP(p)
	if rw {
		be.Reward.Values[0] = 1
	} else {
		be.Reward.Values[0] = 0
	}
	return rw
}

func (be *BanditEnv) Step() bool {
	be.Epoch.Same() // good idea to just reset all non-inner-most counters at start

	if be.RndOpt {
		be.RandomOpt()
	}
	be.SetInput()
	be.SetReward()

	if be.Trial.Incr() {
		be.Epoch.Incr()
	}
	return true
}

func (be *BanditEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (be *BanditEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return be.Run.Query()
	case env.Epoch:
		return be.Epoch.Query()
	case env.Trial:
		return be.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*BanditEnv)(nil)
