// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"cogentcore.org/lab/base/randx"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/etensor/tensor"
)

// BanditEnv simulates an n-armed bandit, where each of n inputs is associated with
// a specific probability of reward.
type BanditEnv struct {

	// name of this environment (Train or Test)
	Name string

	// number of different inputs
	N int

	// probabilities for each option
	P []float32 `display:"inline"`

	// value for reward
	RewVal float32

	// value for non-reward
	NoRewVal float32

	// bandit option current / prev
	Option env.CurPrvInt

	// if true, select option at random each Step -- otherwise must be set externally (e.g., by model)
	RndOpt bool

	// one-hot input representation of current option
	Input tensor.Float64

	// single reward value
	Reward tensor.Float64
}

func (ev *BanditEnv) Label() string { return ev.Name }

// SetN initializes env for given number of options, and inits states
func (ev *BanditEnv) SetN(n int) {
	ev.N = n
	ev.P = make([]float32, n)
	ev.Input.SetShape([]int{n})
	ev.Reward.SetShape([]int{1})
	if ev.RewVal == 0 {
		ev.RewVal = 1
	}
}

func (ev *BanditEnv) State(element string) tensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "SNc":
		return &ev.Reward
	}
	return nil
}

// String returns the current state as a string
func (ev *BanditEnv) String() string {
	return fmt.Sprintf("S_%d_%g", ev.Option.Cur, ev.Reward.Values[0])
}

func (ev *BanditEnv) Init(run int) {
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
	rw := randx.BoolP(float64(p))
	if rw {
		ev.Reward.Values[0] = float64(ev.RewVal)
	} else {
		ev.Reward.Values[0] = float64(ev.NoRewVal)
	}
	return rw
}

func (ev *BanditEnv) Step() bool {
	if ev.RndOpt {
		ev.RandomOpt()
	}
	ev.SetInput()
	ev.SetReward()
	return true
}

func (ev *BanditEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*BanditEnv)(nil)
